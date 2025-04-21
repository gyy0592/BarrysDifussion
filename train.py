# Cell 1: Imports and Configuration
import torch
import yaml
from utils.mytrain import train_diffusion_denoiser_1d
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from utils.network_model import DDPM1D

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Cell 4: Test Model Initialization and Forward Pass


with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
model_config = config['model']
BATCH_SIZE = config['general']['batch_size']
# Set variables from config

BATCH_SIZE = config['general']['batch_size']
LEARNING_RATE = config['general']['learning_rate']
EPOCHS = config['general']['epochs']
TIMESTEPS = config['diffusion']['timesteps']

model = DDPM1D(config).to(device)



dataset_root = Path("dataset")
train_dir = dataset_root / "train"
test_dir = dataset_root / "test"
valid_dir = dataset_root / "valid"
# Cell 3: DataLoader Creation

# --- Create PyTorch Datasets and DataLoaders from saved files ---
train_dataset = TensorDataset(torch.load(train_dir / "clean_signals.pt"))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Test dataset can contain both clean and noisy for evaluation purposes
test_dataset = TensorDataset(
    torch.load(test_dir / "clean_signals.pt"), 
    torch.load(test_dir / "noisy_signals.pt")
)



test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_diffusion_denoiser_1d(model, train_loader, test_loader, device, EPOCHS=EPOCHS, TIMESTEPS=TIMESTEPS, LEARNING_RATE=LEARNING_RATE, config = config)