"""
Main components:
  * Argument parser + JSON config loader for lipreading backbone/TCN.
  * Lipreading model (feature guidance) loaded via `load_model`.
  * SR generator + discriminator (from model) trained with a compound loss:
      - Adversarial (MSE GAN)
      - Content (L1 on VGG features)
      - Multi‑scale MSE terms (64 & 128 spatial size frames)
      - Focal Frequency Loss (FFT domain) on 64 & 128 resolutions
      - DCT reconstruction loss on generator DCT branch
      - Lipreading temporal feature alignment (TCN activation distance)
  * Training + validation loops with learning rate schedulers.
  * Periodic image grid saving & best‑validation checkpointing.

NOTE: This script assumes availability of custom modules:
  model.aud_vid_gen, model.Discriminator, tcn.Lipreading
  dataloader_lip.Dataset
  utils: showLR, calculateNorm2, AverageMeter, load_model, load_json, save2npz
  focal_frequency_loss.FocalFrequencyLoss (FFL)

"""
from __future__ import annotations
import os
import math
import argparse
import random
from pathlib import Path
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as data_utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision.models import vgg19
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

# Third‑party / local modules
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from focal_frequency_loss import FocalFrequencyLoss as FFL
from torchjpeg.dct import batch_dct

from model import Wav2Lip_u, Discriminator,
from tcn import Lipreading
from dataloader_lip import Dataset
from utils import calculateNorm2, load_model, load_json

# ---------------------------------------------------------------------------
# Reproducibility & warnings
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cudnn.benchmark = True  # can speed up with fixed input sizes

# ---------------------------------------------------------------------------
# Argument Handling
# ---------------------------------------------------------------------------

def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Lipreading‑guided SRGAN Training')
    # Dataset / modality
    parser.add_argument('--dataset', default='lrw', help='Dataset selection (unused placeholder)')
    parser.add_argument('--num-classes', type=int, default=9, help='Number of lipreading classes')
    parser.add_argument('--modality', default='video', choices=['video', 'raw_audio'], help='Modality for lipreading model input')

    # Label / annotation paths
    parser.add_argument('--label-path', type=str, default='./labels/500WordsSortedList_test.txt', help='Path to labels txt file (optional)')
    parser.add_argument('--annonation-direc', default=None, help='Annotation directory (optional)')

    # Backbone / TCN config (overridden by JSON config file)
    parser.add_argument('--backbone-type', type=str, default='resnet', choices=['resnet', 'shufflenet'])
    parser.add_argument('--relu-type', type=str, default='relu', choices=['relu','prelu'])
    parser.add_argument('--width-mult', type=float, default=1.0, help='Width multiplier')
    parser.add_argument('--tcn-kernel-size', type=int, nargs='+')
    parser.add_argument('--tcn-num-layers', type=int, default=4)
    parser.add_argument('--tcn-dropout', type=float, default=0.2)
    parser.add_argument('--tcn-dwpw', action='store_true', help='Use depthwise separable conv in TCN')
    parser.add_argument('--tcn-width-mult', type=int, default=1)

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam','sgd','adamw'])
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate (lipreading model, if trained)')
    parser.add_argument('--init-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--test', action='store_true', help='Only run evaluation loop')

    # Pretrained lipreading model
    parser.add_argument('--model-path', type=str, default="/home/dspadmin/Downloads/lrw_resnet18_mstcn.pth.tar", help='Pretrained lipreading model path')
    parser.add_argument('--allow-size-mismatch', action='store_true', default=True, help='Allow size mismatch when loading lipreading model')

    # Feature extraction / embeddings
    parser.add_argument('--extract-feats', action='store_true', help='Return features instead of logits in lipreading model')

    # JSON configuration for lipreading model architecture
    parser.add_argument('--config-path', type=str, default="/home/dspadmin/Downloads/Lipreading_using_Temporal_Convolutional_Networks-master/configs/lrw_resnet18_mstcn.json")

    # Misc
    parser.add_argument('--interval', type=int, default=50, help='Logging interval (batches)')
    parser.add_argument('--workers', type=int, default=8, help='Data loading workers')
    parser.add_argument('--logging-dir', type=str, default='./train_logs')
    parser.add_argument('--queue-length', type=int, nargs='+', default=[5,5,5,5], help='(Unused) placeholder for queue length')

    return parser.parse_args()

args = load_args()

# ---------------------------------------------------------------------------
# Lipreading Model Loader
# ---------------------------------------------------------------------------

def build_lipreading_from_json() -> Lipreading:
    """Load lipreading model architecture & params from JSON config."""
    assert args.config_path.endswith('.json') and os.path.isfile(args.config_path), \
        f"Config JSON not found: {args.config_path}"
    cfg = load_json(args.config_path)

    # Override args with config
    args.backbone_type = cfg['backbone_type']
    args.width_mult    = cfg['width_mult']
    args.relu_type     = cfg['relu_type']

    tcn_options = {
        'num_layers':   cfg['tcn_num_layers'],
        'kernel_size':  cfg['tcn_kernel_size'],
        'dropout':      cfg['tcn_dropout'],
        'dwpw':         cfg['tcn_dwpw'],
        'width_mult':   cfg['tcn_width_mult'],
    }

    model = Lipreading(
        modality=args.modality,
        num_classes=args.num_classes,
        tcn_options=tcn_options,
        backbone_type=args.backbone_type,
        relu_type=args.relu_type,
        width_mult=args.width_mult,
        extract_feats=args.extract_feats
    ).cuda()

    calculateNorm2(model)  # Diagnostic printout of parameter norms
    return model

lipnet = build_lipreading_from_json()
lipnet = load_model(args.model_path, lipnet, allow_size_mismatch=args.allow_size_mismatch)

# ---------------------------------------------------------------------------
# Hook for TCN feature extraction (temporal guidance loss)
# ---------------------------------------------------------------------------
activation: Dict[str, torch.Tensor] = {}

def get_activation(name: str):
    def hook(_model, _input, output):
        activation[name] = output.detach()
    return hook

lipnet.tcn.mb_ms_tcn.register_forward_hook(get_activation("mb_ms_tcn"))

# ---------------------------------------------------------------------------
# Perceptual (VGG) + Style Loss Module
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """Perceptual + style loss using selected VGG layers."""
    def __init__(self,
                 layer_weights: Dict[str, float],
                 vgg_type: str = 'vgg19',
                 use_input_norm: bool = True,
                 range_norm: bool = False,
                 perceptual_weight: float = 1.0,
                 style_weight: float = 50.0,
                 criterion: str = 'l1'):
        super().__init__()
        self.layer_weights = layer_weights
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm
        )
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        elif criterion == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'Unsupported criterion: {criterion}')
        self.criterion_type = criterion

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        x_feats = self.vgg(x)
        gt_feats = self.vgg(gt.detach())

        percep_loss = None
        if self.perceptual_weight > 0:
            l = 0
            for k in x_feats.keys():
                diff = x_feats[k] - gt_feats[k]
                if self.criterion_type == 'fro':
                    l += torch.norm(diff, p='fro') * self.layer_weights[k]
                else:
                    l += self.criterion(x_feats[k], gt_feats[k]) * self.layer_weights[k]
            percep_loss = l * self.perceptual_weight

        style_loss = None
        if self.style_weight > 0:
            s = 0
            for k in x_feats.keys():
                gx = self._gram(x_feats[k])
                ggt = self._gram(gt_feats[k])
                if self.criterion_type == 'fro':
                    s += torch.norm(gx - ggt, p='fro') * self.layer_weights[k]
                else:
                    s += self.criterion(gx, ggt) * self.layer_weights[k]
            style_loss = s * self.style_weight
        return percep_loss, style_loss

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        n, c, h, w = feat.size()
        f = feat.view(n, c, -1)
        return f.bmm(f.transpose(1, 2)) / (c * h * w)

percep_layer_weights = {
    'conv1_2': 0.1,
    'conv2_2': 0.1,
    'conv3_4': 1.0,
    'conv4_4': 1.0,
    'conv5_4': 1.0
}
percep_loss_fn = PerceptualLoss(percep_layer_weights).cuda()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BATCH_SIZE = 2  
EPOCHS = args.epochs

train_dataset = Dataset('train')
val_dataset   = Dataset('val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, drop_last=True)

# ---------------------------------------------------------------------------
# Models (Generator / Discriminator / VGG feature extractor)
# ---------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """Slice of VGG19 up to conv3_4 (layer index 18) for content loss."""
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        self.extractor = nn.Sequential(*list(vgg.features.children())[:18])
        for p in self.extractor.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

feature_extractor = FeatureExtractor().cuda().eval()

# SRGAN components
generator = aud_vid_gen()
discriminator = Discriminator(input_shape=(3, 128, 128))

# Parallel (if multiple GPUs available)
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

generator = MyDataParallel(generator, device_ids=[0,1])
discriminator = MyDataParallel(discriminator, device_ids=[0,1])

generator.cuda(); discriminator.cuda()

# ---------------------------------------------------------------------------
# Loss Functions & Utilities
# ---------------------------------------------------------------------------
criterion_gan      = nn.MSELoss().cuda()
criterion_content  = nn.L1Loss().cuda()
ffl                = FFL(loss_weight=1.0, alpha_ffl=1.0).cuda()  # Focal Frequency Loss
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# DCT transform (wrapped for transforms.Compose style usage)

def dct_fun(t: torch.Tensor) -> torch.Tensor:
    return batch_dct(t)

bc_transform = transforms.Compose([transforms.Lambda(lambda x: dct_fun(x))])

# ---------------------------------------------------------------------------
# Optimizers & LR Schedulers
# ---------------------------------------------------------------------------
LR = 1e-5
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.999))

scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=2, gamma=0.1)
scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=2, gamma=0.1)

# ---------------------------------------------------------------------------
# Training / Validation Loops
# ---------------------------------------------------------------------------
train_gen_losses: List[float] = []
train_disc_losses: List[float] = []
val_gen_losses: List[float] = []
val_disc_losses: List[float] = []

lengths_s = [5,5]  # Sequence length meta info for lipnet forward (placeholder)

def flatten_time(x: torch.Tensor) -> torch.Tensor:
    """Concatenate temporal dimension onto batch (expects shape B x C x T x H x W)."""
    return torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)

os.makedirs("images_srgan", exist_ok=True)
os.makedirs("saved_models_srgan", exist_ok=True)

best_val_gen = math.inf

for epoch in range(EPOCHS):
    # ------------------------- TRAIN -------------------------
    generator.train(); discriminator.train(); lipnet.eval()
    gen_loss_epoch = 0.0
    disc_loss_epoch = 0.0

    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}', total=len(train_loader))
    for batch_idx, (y_low, indiv_mels, mel, gt, gt64) in enumerate(pbar):
        y_low      = y_low.type(Tensor)
        indiv_mels = indiv_mels.type(Tensor)
        gt         = gt.type(Tensor)
        gt64       = gt64.type(Tensor)

        gt_dct = flatten_time(gt)
        gt64_f = flatten_time(gt64)

        valid = Variable(Tensor(np.ones((gt_dct.size(0), *discriminator.output_shape))), requires_grad=False)
        fake  = Variable(Tensor(np.zeros((gt_dct.size(0), *discriminator.output_shape))), requires_grad=False)

        # ---- Generator ----
        optimizer_G.zero_grad()
        gen_hr_seq, gen_hr64_seq, gen_dct = generator(indiv_mels, y_low)
        gen_hr   = flatten_time(gen_hr_seq)
        gen_hr64 = flatten_time(gen_hr64_seq)

        # Adversarial loss
        loss_gan = criterion_gan(discriminator(gen_hr), valid)

        # Content (VGG) loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(gt_dct)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Multi‑scale MSE losses
        loss_mse64  = criterion_gan(gen_hr64, gt64_f)
        loss_mse128 = criterion_gan(gen_hr, gt_dct)

        # Frequency domain (FFL) losses
        loss_fft64  = ffl(gen_hr64, gt64_f)
        loss_fft128 = ffl(gen_hr, gt_dct)
        loss_fft = loss_fft64 + loss_fft128

        # DCT branch loss
        gt_features_dct = bc_transform(gt_dct)
        loss_dct = criterion_content(gen_dct, gt_features_dct.detach())

        # Lipreading feature alignment (temporal crop center region 64:128) -> pass through lipnet
        gen_crop = gen_hr_seq[:, :, :, 64:128, :][:,0].unsqueeze(1)  # shape B x 1 x T x H x W
        gt_crop  = gt[:, :, :, 64:128, :][:,0].unsqueeze(1)
        _ = lipnet(gen_crop, lengths=lengths_s); gen_tcn = activation['mb_ms_tcn']
        _ = lipnet(gt_crop, lengths=lengths_s);  gt_tcn  = activation['mb_ms_tcn']
        loss_lip = criterion_gan(gen_tcn, gt_tcn)

        # Aggregate generator loss (weights tuned empirically)
        loss_G = (loss_content + 1e-3 * loss_gan + 0.0001 * loss_dct +
                  0.001 * loss_mse64 + 0.01 * loss_mse128 + 0.0001 * loss_fft + 0.0001 * loss_lip)
        loss_G.backward()
        optimizer_G.step()

        # ---- Discriminator ----
        optimizer_D.zero_grad()
        loss_fake = criterion_gan(discriminator(gen_hr.detach()), fake)
        loss_real = criterion_gan(discriminator(gt_dct), valid)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        gen_loss_epoch += loss_G.item()
        disc_loss_epoch += loss_D.item()
        train_gen_losses.append(loss_G.item())
        train_disc_losses.append(loss_D.item())
        pbar.set_postfix(G=gen_loss_epoch/(batch_idx+1), D=disc_loss_epoch/(batch_idx+1))

    # ------------------------- VALIDATION -------------------------
    generator.eval(); discriminator.eval()
    val_gen_epoch = 0.0
    val_disc_epoch = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Val   Epoch {epoch}', total=len(val_loader))
        for batch_idx, (y_low, indiv_mels, mel, gt, gt64) in enumerate(pbar):
            y_low      = y_low.type(Tensor)
            indiv_mels = indiv_mels.type(Tensor)
            gt         = gt.type(Tensor)
            gt64       = gt64.type(Tensor)

            gt_dct = flatten_time(gt)
            gt64_f = flatten_time(gt64)

            valid = Variable(Tensor(np.ones((gt_dct.size(0), *discriminator.output_shape))), requires_grad=False)
            fake  = Variable(Tensor(np.zeros((gt_dct.size(0), *discriminator.output_shape))), requires_grad=False)

            gen_hr_seq, gen_hr64_seq, gen_dct = generator(indiv_mels, y_low)
            gen_hr   = flatten_time(gen_hr_seq)
            gen_hr64 = flatten_time(gen_hr64_seq)

            loss_gan = criterion_gan(discriminator(gen_hr), valid)
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(gt_dct)
            loss_content = criterion_content(gen_features, real_features.detach())
            loss_mse64  = criterion_gan(gen_hr64, gt64_f)
            loss_mse128 = criterion_gan(gen_hr, gt_dct)
            loss_fft64  = ffl(gen_hr64, gt64_f)
            loss_fft128 = ffl(gen_hr, gt_dct)
            loss_fft = loss_fft64 + loss_fft128
            gt_features_dct = bc_transform(gt_dct)
            loss_dct = criterion_content(gen_dct, gt_features_dct.detach())

            gen_crop = gen_hr_seq[:, :, :, 64:128, :][:,0].unsqueeze(1)
            gt_crop  = gt[:, :, :, 64:128, :][:,0].unsqueeze(1)
            _ = lipnet(gen_crop, lengths=lengths_s); gen_tcn = activation['mb_ms_tcn']
            _ = lipnet(gt_crop, lengths=lengths_s);  gt_tcn  = activation['mb_ms_tcn']
            loss_lip = criterion_gan(gen_tcn, gt_tcn)

            loss_G = (loss_content + 1e-3 * loss_gan + 0.0001 * loss_dct +
                      0.001 * loss_mse64 + 0.01 * loss_mse128 + 0.0001 * loss_fft + 0.0001 * loss_lip)

            loss_fake = criterion_gan(discriminator(gen_hr.detach()), fake)
            loss_real = criterion_gan(discriminator(gt_dct), valid)
            loss_D = 0.5 * (loss_real + loss_fake)

            val_gen_epoch += loss_G.item()
            val_disc_epoch += loss_D.item()
            pbar.set_postfix(G=val_gen_epoch/(batch_idx+1), D=val_disc_epoch/(batch_idx+1))

            # Periodically save sample grid (GT | upsampled LR | SR)
            if random.random() < 0.1:
                y_low_flat = flatten_time(y_low)
                imgs_lr = F.interpolate(y_low_flat, scale_factor=4)
                imgs_hr = make_grid(gt_dct, nrow=1, normalize=True)
                gen_hr_grid = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr_grid = make_grid(imgs_lr, nrow=1, normalize=True)
                grid = torch.cat((imgs_hr, imgs_lr_grid, gen_hr_grid), -1)
                save_image(grid, f"images_srgan/epoch{epoch}_batch{batch_idx}.png", normalize=False)

    scheduler_G.step()
    scheduler_D.step()

    val_gen_mean = val_gen_epoch / len(val_loader)
    val_disc_mean = val_disc_epoch / len(val_loader)
    val_gen_losses.append(val_gen_mean)
    val_disc_losses.append(val_disc_mean)

    # Checkpoint best generator (by validation generator loss)
    if val_gen_mean < best_val_gen:
        best_val_gen = val_gen_mean
        torch.save(generator.module.state_dict(), 'saved_models_srgan/generator_best.pth')
        torch.save(discriminator.module.state_dict(), 'saved_models_srgan/discriminator_best.pth')
        print(f"[Checkpoint] Epoch {epoch}: Saved new best models (val G loss={best_val_gen:.4f})")

print('Training complete.')

