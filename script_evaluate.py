import os
import yaml
import torch
from utils.network_model import DDPM1D
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from utils.myeval import evaluate_model  # Import the evaluation function


# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
BATCH_SIZE = config['general']['batch_size']
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = DDPM1D(config).to(device)

# Load best model weights
best_model_path = "output/best_diffusion_denoiser_1d.pth"
if os.path.exists(best_model_path):
    print(f"Loading best model weights from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
else:
    print("Warning: Best model checkpoint not found. Using current model state.")

model.eval()

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
# Run evaluation using the imported function
mse_noisy, mse_denoised, improvement = evaluate_model(model, test_loader, config, device)

print(f"Evaluation complete with {improvement:.2f}% MSE improvement")