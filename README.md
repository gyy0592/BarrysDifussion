# 1D Signal Denoising with Diffusion Models

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for cleaning 1D signals (like time series data) using PyTorch. The implementation focuses on noise removal from synthetic sine waves, but can be adapted to other 1D signal types.

## Architecture

The model uses a U-Net style architecture adapted for 1D signals with time-step conditioning:

```
Input Signal
    ↓
Level 0 (channels=64×1=64)  ← Each level has num_res_blocks residual blocks
    ↓ Downsample 2×
Level 1 (channels=64×2=128) ← Each level has num_res_blocks residual blocks
    ↓ Downsample 2×
Level 2 (channels=64×4=256) ← attention_levels includes 2, so attention applied here (*)
    ↓ Downsample 2×
Level 3 (channels=64×8=512) ← attention_levels includes 3, so attention applied here (*)
    ↓
Bottleneck (512 channels)   ← use_attention enables attention here
    ↓
Upsample + Level 3 skip    ← attention_levels includes 3, so attention applied here (*)
    ↓
Upsample + Level 2 skip    ← attention_levels includes 2, so attention applied here (*)
    ↓
Upsample + Level 1 skip
    ↓
Upsample + Level 0 skip
    ↓
Output Signal
```
(*) Note: Multi-level attention is configured but not currently implemented in the code.

### Key Components

1. **Time Embeddings**: Uses sinusoidal position encodings to embed diffusion timesteps.
2. **Residual Blocks**: Incorporate time embeddings and maintain residual connections for better gradient flow.
3. **Self-Attention**: Captures long-range dependencies in the signal (currently at bottleneck only).
4. **GroupNorm**: Used throughout the network with `min_groups` parameter controlling the number of groups.
5. **Direct Path**: Signal preservation mechanism with configurable scaling factor.

## Configuration

The model is configured through a YAML file with the following key parameters:

```yaml
model:
  in_channels: 1               # Number of input channels
  out_channels: 1              # Number of output channels
  time_dim: 1024               # Time embedding dimension
  base_channels: 64            # Base channels
  channel_mults: [1, 2, 4, 8]  # Channel multipliers (defines network depth)
  use_attention: true          # Use attention in bottleneck
  multi_level_attention: true  # Attention at multiple levels
  attention_levels: [2, 3]     # Apply attention at these levels
  kernel_size: 5               # Kernel size for convolutions
  direct_path_scale: 0.2       # Direct path scale for signal preservation
  min_groups: 8                # Minimum groups for GroupNorm layers
  num_res_blocks: 3            # Number of residual blocks per level
  dropout_rate: 0.3            # Dropout rate
```

## Dataset

The project generates synthetic sine wave data with configurable:
- Number of samples for training and testing
- Signal length
- Noise standard deviation
- Varying parameters (amplitude, frequency, phase)

## Training Process

Training implements:
- AdamW optimizer with weight decay
- Cosine annealing learning rate scheduler
- Early stopping based on validation loss
- Gradient clipping for stability
- Comprehensive CSV logging and loss visualization

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- tqdm
- PyYAML
- ......

### Installation

In todo list yet.

### Usage
<!--  we have script_dataset_generation.py  script_evaluate.py  script_train.py -->
1. Configure the model in `config/config.yaml`
2. Generate dataset:
   ```python
   python script_dataset_generation.py
   ```
3. Train the model:
   ```python
python script_train.py
   ```
4. Evaluate the model:
   ```python
    python script_evaluate.py
   ```

## Implementation Notes

There are a few discrepancies between the current configuration and implementation:

1. The model uses 2 residual blocks per level, while the config specifies 3
2. Multi-level attention is configured but not implemented yet
3. Dropout is in the config but not used in the model

## Results

Training results are saved in the `output` directory, including:
- Best model weights
- Training logs in CSV format
- Loss curve visualization