# Audio-Visual Video Face Hallucination with Lipreading and Frequency-Aware Loss

This repository contains the official code for our proposed **audio-visual cross-modality support based video face hallucination network** (https://link.springer.com/article/10.1007/s00138-025-01699-4), designed to enhance facial details in low-resolution videos. Our model leverages audio cues and a lipreading component to enhance spatial detail and maintain temporal consistency across frames.

---

## Key Features

- **Cross-Modal Learning**: Leverages correlations between facial motion and speech signals.
- **Lipreading-Aware Loss**: Guides reconstruction with fine-grained motion cues from lip movements.
- **Focal Frequency Loss**: Enhances high-frequency components often missed in GAN-based approaches.
- **Temporal Consistency**: Improves coherence in generated video sequences.

---

## Project Structure

- `train.py`: Main training script.
- `model.py`: Audio-visual hallucination network architecture.
- `tcn.py`: Lipreading Temporal Convolutional Network.
- `focal_frequency_loss.py`: Frequency-aware loss function.
- `dataloader.py`: Video/audio dataset loader.
- `utils.py`: Utility functions.
- `environment.yml`: Conda environment specification.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate mva_env
```

---

## Training

Run the training script with the data:


python train.py --data_root /path/to/data_folder


---

## Acknowledgements


| TCN Lipreading Model | https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks 
| Focal Frequency Loss | https://github.com/EndlessSora/focal-frequency-loss
