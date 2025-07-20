#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ──────────────────────────────────────────────────────────────────────────────
# Common helpers
# ──────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial
from typing import Any, List, Sequence, Tuple
from self_attention_cv import AxialAttentionBlock
from torchjpeg.dct import batch_dct, block_idct, block_dct, fdct, idct
from torchjpeg.dct import batch_idct

#%%
def dct_fun(ten):
    return batch_dct(ten)

def idct_fun(ten):
    return batch_idct(ten)
def block_dt(ten):
    return fdct(ten)
def block_idt(ten):
    return idct(ten)

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

# reusable conv blocks --------------------------------------------------------
def conv_bn_relu(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1,
                 residual: bool = False) -> nn.Sequential:
    """Conv → BN → ReLU wrapper (optionally residual)"""
    return nn.Sequential(
        Conv2d(cin, cout, k, s, p, residual=residual)  # keeps residual logic
    )

def conv_lrelu(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1,
               negative_slope: float = 0.01) -> nn.Sequential:
    """Plain Conv2d → LeakyReLU block (no BN, no residual)"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p),
        nn.LeakyReLU(negative_slope, inplace=True)
    )
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
       # nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #nn.ReLU(inplace=True)
    )


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

def deconv_bn_relu(cin: int, cout: int, k: int, s: int, p: int,
                   out_pad: int = 0) -> nn.Sequential:
    """ConvTranspose → BN → ReLU."""
    return nn.Sequential(
        Conv2dTranspose(cin, cout, k, s, p, output_padding=out_pad)
    )

# up-sample block used twice ---------------------------------------------------
def pixelshuffle_x2(cin: int) -> nn.Sequential:
    """Conv->PixelShuffle(x2)->PReLU exactly as in the original code."""
    return nn.Sequential(
        nn.Conv2d(cin, 4 * cin, kernel_size=3, stride=1, padding=1),
        nn.PixelShuffle(2),
        nn.PReLU()
    )
# ──────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
  """Squeeze-and-excitation block"""
  def __init__(self, n_in, r=24):
    super().__init__()

    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
                                    nn.SiLU(),
                                    nn.Conv2d(n_in//r, n_in, kernel_size=1),
                                    nn.Sigmoid())

  def forward(self, x):
    y = self.squeeze(x)
    y = self.excitation(y)
    return x * y
#%%
class DropSample(nn.Module):
  """Drops each sample in x with probability p during training"""
  def __init__(self, p=0):
    super().__init__()

    self.p = p

  def forward(self, x):
    if (not self.p) or (not self.training):
      return x

    batch_size = len(x)
    random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
    bit_mask = self.p<random_tensor

    x = x.div(1-self.p)
    x = x * bit_mask
    return x
#%%
#%%
class ConvBnAct(nn.Module):
  """Layer grouping a convolution, batchnorm, and activation function"""
  def __init__(self, n_in, n_out, kernel_size=3,
               stride=1, padding=0, groups=1, bias=False,
               bn=True, act=True):
    super().__init__()

    self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                          stride=stride, padding=padding,
                          groups=groups, bias=bias)
    #self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
    self.act = nn.SiLU() if act else nn.Identity()

  def forward(self, x):
    x = self.conv(x)
    #x = self.bn(x)
    x = self.act(x)
    return x

class MBConvN(nn.Module):
  """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
  def __init__(self, n_in, n_out, expansion_factor,
               kernel_size=3, stride=1, r=24, p=0):
    super().__init__()

    padding = (kernel_size-1)//2
    expanded = expansion_factor*n_in
    self.skip_connection = (n_in == n_out) and (stride == 1)

    self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded, kernel_size=1)
    self.depthwise = ConvBnAct(expanded, expanded, kernel_size=kernel_size,
                               stride=stride, padding=padding, groups=expanded)
    self.se = SEBlock(expanded, r=r)
    self.reduce_pw = ConvBnAct(expanded, n_out, kernel_size=1,
                               act=False)
    self.dropsample = DropSample(p)

  def forward(self, x):
    residual = x

    x = self.expand_pw(x)
    x = self.depthwise(x)
    x = self.se(x)
    x = self.reduce_pw(x)

    if self.skip_connection:
      x = self.dropsample(x)
      x = x + residual

    return x
class InceptionA(nn.Module):

    def __init__(self, c_in, c_red : dict, c_out : dict, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            #nn.BatchNorm2d(c_out["1x1"]),
            nn.SiLU()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
           # nn.BatchNorm2d(c_red["3x3"]),
            nn.SiLU(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
           # nn.BatchNorm2d(c_out["3x3"]),
            nn.SiLU()
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
           # nn.BatchNorm2d(c_red["5x5"]),
            nn.SiLU(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
          #  nn.BatchNorm2d(c_out["5x5"]),
            nn.SiLU()
        )

        # Max-pool branch



    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
       # x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        return x_out

#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class aud_vid_gen(nn.Module):
    def __init__(self):
        super().__init__()

        # ───── 1. video/image encoder (“c*” blocks) ──────────────────────────
        c_specs: Sequence[Tuple[str, int, int, int, int, int, bool]] = [
            # name   cin cout k s p residual
            ("c0",     3,   3, 3, 1, 1, False),
            ("c0_0",   3,   3, 3, 2, 1, False),
            ("c1",     3,  16, 3, 1, 1, False),
            ("c2",    16,  32, 3, 2, 1, False),
            ("c3",    32,  32, 3, 1, 1, True ),
            ("c4",    32,  32, 3, 1, 1, True ),
            ("c5",    32,  64, 3, 2, 1, False),
            ("c6",    64,  64, 3, 1, 1, True ),
            ("c7",    64,  64, 3, 1, 1, True ),
            ("c8",    64,  64, 3, 1, 1, True ),
            ("c9",    64, 128, 3, 2, 1, False),
            ("c10",  128, 128, 3, 1, 1, True ),
            ("c11",  128, 128, 3, 1, 1, True ),
            ("c12",  128, 256, 3, 2, 1, False),
            ("c13",  256, 256, 3, 1, 1, True ),
            ("c14",  256, 256, 3, 1, 1, True ),
        ]
        for n, ci, co, k, s, p, res in c_specs:
            setattr(self, n, conv_bn_relu(ci, co, k, s, p, residual=res))
        self.c15 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)

        # ───── 2. audio encoder (“audio_encoder*”) ───────────────────────────
        ae_specs: Sequence[Tuple[str, int, int, int, Tuple[int, int], int, bool]] = [
            # name   cin cout k stride     p residual
            ("ae0",   1,  32, 3, (1, 1),   1, False),
            ("ae1",  32,  32, 3, (1, 1),   1, True ),
            ("ae2",  32,  32, 3, (1, 1),   1, True ),
            ("ae3",  32,  64, 3, (3, 1),   1, False),
            ("ae4",  64,  64, 3, (1, 1),   1, True ),
            ("ae5",  64,  64, 3, (1, 1),   1, True ),
            #("ae6",  64, 128, 3, (3, 1),   1, False),
            ("ae6", 64, 128, 3, (3, 3), 1, False),
            ("ae7", 128, 128, 3, (1, 1),   1, True ),
            ("ae8", 128, 128, 3, (1, 1),   1, True ),
            ("ae9", 128, 256, 3, (3, 2),   1, False),
            ("ae10",256, 256, 3, (1, 1),   1, True ),
            ("ae11",256, 256, 3, (1, 1),   1, True ),
        ]
        for n, ci, co, k, st, p, res in ae_specs:
            setattr(self, f"audio_encoder{n[2:]}", conv_bn_relu(ci, co, k, st, p, residual=res))
        self.audio_encoder = self.audio_encoder0
        self.audio_encoder12 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0)

        # ───── 3. decoder blocks (transpose-conv heavy) ──────────────────────
        d_specs: Sequence[Tuple[int, int, int, int, int, int, bool]] = [
            # cin cout k  s  p out_pad  residual
            (512, 512, 1, 1, 0, 0,  False),   # d1
            (512, 512, 5, 2, 2, 1,  True ),   # d2
            (512, 512, 3, 1, 1, 0,  True ),   # d3
            (512, 512, 3, 2, 1, 1,  True ),   # d4
            (512, 512, 3, 1, 1, 0,  True ),   # d5
            (512, 512, 3, 1, 1, 0,  True ),   # d6
            (512, 384, 3, 2, 1, 1,  True ),   # d7
            (384, 384, 3, 1, 1, 0,  True ),   # d8
            (384, 384, 3, 1, 1, 0,  True ),   # d9
            (384, 256, 3, 2, 1, 1,  True ),   # d10
            (256, 256, 3, 1, 1, 0,  True ),   # d11
            (256, 256, 3, 1, 1, 0,  True ),   # d12
            (256, 128, 3, 2, 1, 1,  True ),   # d13
            (128, 128, 3, 1, 1, 0,  True ),   # d14
            (128, 128, 3, 1, 1, 0,  True ),   # d15
            (128,  64, 3, 2, 1, 1,  True ),   # d16
            ( 64,  64, 3, 1, 1, 0,  True ),   # d17
            ( 64,  64, 3, 1, 1, 0,  True ),   # d18
        ]
        # helper to pick conv or deconv block
        def make_dec(cin, cout, k, s, p, op, res):
            return deconv_bn_relu(cin, cout, k, s, p, op) if s > 1 \
                   else conv_bn_relu(cin, cout, k, s, p, residual=res)

        self.decoder_blocks = nn.ModuleList([make_dec(*spec) for spec in d_specs])

        # ───── 4. misc heads / conv stubs (unchanged) ────────────────────────
        self.dconv_down4 = double_conv(256, 512)

        self.output_block  = nn.Sequential(
            conv_bn_relu(64, 32, 3, 1, 1),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.output_block1 = nn.Sequential(
            conv_bn_relu(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid(),
        )

        #self.bottleneck_block = BottleneckBlock(128, (64, 64), heads=4,out_channels=128, pooling=True)
        self.att = AxialAttentionBlock(128, dim=32, heads=8)
        self.output_block1 = nn.Sequential(           # 64 → 128 → 128 (sigmoid)
            conv_bn_relu(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv_3  = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)  # keeps channels
        self.conv_v1 = nn.Conv2d(128,  32,  kernel_size=3, padding=1)  # 128 → 32
        self.conv_v2 = nn.Conv2d( 32,   3,  kernel_size=3, padding=1)  # 32  → 3
        # small pre-/post-net convs
        self.conv_u1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv_u2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_u3 = nn.Conv2d(64, 3, 3, padding=1)
        self.conv6 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1))

        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)

        self.conv3fs = nn.Sequential(          # final tanh head
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Tanh(),
        )

        # pixel-shuffle ×2
        self.upsampling = pixelshuffle_x2(64)

        # U-Net style blocks
        self.down_convs = nn.ModuleList([
            double_conv(  3,  64),
            double_conv( 64, 128),
            double_conv(128, 256),
            double_conv(256, 512),
        ])
        self.up_convs = nn.ModuleList([
            double_conv(256 + 512, 256),
            double_conv(128 + 256, 128),
            double_conv( 64 + 128,  64),
        ])

        self.maxpool  = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)

        
        self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4  = self.down_convs
        self.dconv_up3,   self.dconv_up2,   self.dconv_up1   = self.up_convs

        self.inc_blocks1 = nn.Sequential(
            InceptionA(64,
                       c_red={"3x3": 64, "5x5": 64},
                       c_out={"1x1": 64, "3x3": 64, "5x5": 64},
                       act_fn=nn.LeakyReLU)
        )
        self.conv_2 = nn.Conv2d(192, 64, kernel_size=3, padding=1)

        # reused second Inception block 
        self.inc_blocks2 = nn.Sequential(
            InceptionA(64,
                       c_red={"3x3": 64, "5x5": 64},
                       c_out={"1x1": 64, "3x3": 64, "5x5": 64},
                       act_fn=nn.LeakyReLU)
        )

        # projects 64-ch map to 3-ch to match x2
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        # helper: one Inception-residual cycle ---------------------------------
        def _inc_cycle(t: Tensor, last: bool = False) -> tuple[Tensor, Tensor]:
            a = self.inc_blocks1(t)                  # out2*
            b = self.conv_2(a)                      # out3*
            c = self.inc_blocks1(b)                 # out4*
            d = a + c                               # out5*
            e = self.conv6(d) if last else self.conv_2(d)  # out6* / out6d/j
            return e, b
        self.mb_block1 = nn.Sequential(MBConvN(n_in=256, n_out=64, expansion_factor=3,
               kernel_size=3, stride=1, r=24, p=0))
    # ─────────────────────── forward (encoder/decoder slice) ───────────────────────
    def forward(self, audio_sequences, face_sequences):
        # temporal flatten (unchanged)
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if face_sequences.dim() > 4:  # (B, T, C, H, W)
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], 0
            )
            face_sequences = torch.cat(
                [face_sequences[:, :, i] for i in range(face_sequences.size(2))], 0
            )
        def _inc_cycle(t: Tensor, last: bool = False) -> tuple[Tensor, Tensor]:
            a = self.inc_blocks1(t)                 # InceptionA
            b = self.conv_2(a)
            c = self.inc_blocks1(b)
            d = a + c                               # residual add
            e = self.conv6(d) if last else self.conv_2(d)
            return e, b                             # new tensor, plus out3* (b)

        # 1) conv-shuffle-conv → DCT
        x = self.conv_u1(face_sequences)
        x = self.upsampling(x)
        x = self.conv_u2(x)
        x = self.upsampling(x)
        x = self.conv_u3(x)
        x = batch_dct(x)

        # 2) U-Net encoder
        skips: List[Tensor] = []
        for enc in self.down_convs[:-1]:
            x = enc(x); skips.append(x); x = self.maxpool(x)
        x = self.dconv_down4(x)  # bottom

        # 3) U-Net decoder
        for dec, skip in zip(self.up_convs, reversed(skips)):
            x = self.upsample(x)
            x = torch.cat([x, skip], 1)
            x = dec(x)

        # 4) DCT → RGB
        out_dct = self.conv_last(x)
        x_rgb   = idct_fun(out_dct)
        x_out =  self.conv3fs(x_rgb)
        # ── 5) *Image encoder* path through c0 … c15 ──────────────────────────────
        img = face_sequences
        for name in [
            "c0", "c0_0", "c1", "c2", "c3", "c4", "c5", "c6",
            "c7", "c8", "c9", "c10", "c11", "c12", "c13", "c14"
        ]:
            img = getattr(self, name)(img)
        img = self.c15(img)                              # out1 feature-map
        print('vid', img.size())
        # ── 3. audio-encoder path (audio_encoder0 … 12) ─────────────────
        ae = audio_sequences
        for i in range(12):                              # 0 → 11
            ae = getattr(self, f"audio_encoder{i}")(ae)
        ae = self.audio_encoder12(ae)                    # final embedding
                # final audio embedding
        print('audio', ae.size())
        # ── 6) fuse and decode ----------------------------------------------------
        dec = torch.add(img, ae)                       # torch.add(out1, audio_embedding)
        for blk in self.decoder_blocks:        # decoder_block1 … decoder_block18
            dec = blk(dec)
        dec_embedding = dec

        x2 = self.output_block1(dec_embedding)   # 64-ch → 128-ch
        x2 = self.conv_3(x2)                   # local 3×3
        x2 = self.att(x2)                        # axial attention
        x2 = self.conv_v1(x2)                    # 128 → 32
        x2 = self.conv_v2(x2)                    # 32  → 3


        # ── 8) Inception fusion branch ----------------------------------------
        co1   = self.conv_u1(face_sequences)     # 3 → 64
        out2q = self.inc_blocks1(co1)            # 64 → 192
        out3q = self.conv_2(out2q)               # 192 → 64
        out4q = self.inc_blocks1(out3q)          # reuse same block
        out5q = out4q + out2q                    # residual add
        out6q = self.conv_2(out5q)               # 192 → 64
        con1  = self.conv3(out6q)                # 64 → 3  (tanh)

        out32 = con1 + x2                        # fuse with attention path

        # ── 9) restore temporal dimension, if needed
        if input_dim_size > 4:                   # came from (B, T, …)
            chunks     = torch.split(out32, B, dim=0)  # tuple of length T
            out_sr32   = torch.stack(chunks, dim=2)    # (B, C, T, H, W)
        else:
            out_sr32   = out32                   # (B, C, H, W)

        # ── 7) axial-attention refinement ------------------------------------
        x2 = self.output_block1(dec_embedding)
        x2 = self.conv_3(x2)
        x2 = self.att(x2)
        x2 = self.conv_v1(x2)
        x2 = self.conv_v2(x2)

        # ── 8) first Inception cascade (co1 … out32) -------------------------
        co1   = self.conv_u1(out32)          # out32 came from earlier fusion
        co1   = self.conv_u2(co1)

        t = co1
        for _ in range(4):                   # four regular cycles
            t, _ = _inc_cycle(t)
        t, _   = _inc_cycle(t, last=True)    # fifth cycle with conv6
        mb1    = self.mb_block1(t)

        oute   = co1 + mb1
        oute   = self.upsampling(oute)       # ×2
        out64  = self.conv3(oute)

        if input_dim_size > 4:
            out_sr64 = torch.stack(torch.split(out64, B, 0), 2)
        else:
            out_sr64 = out64

        # ── 9) second Inception cascade (starts from out64) ------------------
        t      = self.conv_u1(out64)
        t      = self.conv_u2(t)

        t, b_first = _inc_cycle(t)           # keep out3f (called b_first here)
        for _ in range(3):                   # three more regular cycles
            t, _ = _inc_cycle(t)
        t, _   = _inc_cycle(t, last=True)    # final cycle with conv6
        mb2    = self.mb_block1(t)

        out    = b_first + mb2               # out3f + out_mb1
        out    = self.upsampling(out)
        out_sr = self.conv_u3(out) + x_out   # add earlier IDCT output

        if input_dim_size > 4:
            outputs_sr = torch.stack(torch.split(out_sr, B, 0), 2)
        else:
            outputs_sr = out_sr

        
        return outputs_sr,out_sr64,x_out 


#Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
