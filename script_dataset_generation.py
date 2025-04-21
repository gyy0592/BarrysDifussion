# Cell 2: Dataset Generation
import numpy as np
import torch
from tqdm import tqdm
import yaml
from pathlib import Path

# Load configuration if not already loaded

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
dataset_config = config['dataset']
general_config = config['general']

# Configuration parameters from YAML
N_SAMPLES_TRAIN = dataset_config['n_samples_train']
N_SAMPLES_TEST = dataset_config['n_samples_test']
SIGNAL_LENGTH = general_config['signal_length']
NOISE_STD_DEV = dataset_config['noise_std_dev']
BATCH_SIZE = general_config['batch_size']

# Create directories for dataset storage
dataset_root = Path("dataset")
train_dir = dataset_root / "train"
test_dir = dataset_root / "test"
valid_dir = dataset_root / "valid"

# Create all directories
for dir_path in [train_dir, test_dir, valid_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

def generate_sine_wave_dataset(n_samples, signal_length, noise_std_dev):
    """
    Generates a dataset of clean sine waves and their noisy versions.
    Args:
        n_samples (int): Number of samples to generate.
        signal_length (int): Length of each time series signal.
        noise_std_dev (float): Standard deviation of the Gaussian noise added.
    Returns:
        tuple: A tuple containing two torch tensors:
               - clean_signals (torch.Tensor): Tensor of shape [n_samples, 1, signal_length]
               - noisy_signals (torch.Tensor): Tensor of shape [n_samples, 1, signal_length]
    """
    clean_signals_list = []
    noisy_signals_list = []
    time = np.linspace(0, 1, signal_length) # Time vector from 0 to 1
    print(f"Generating {n_samples} samples...")
    for _ in tqdm(range(n_samples)):
        # Randomize parameters for each sine wave
        amplitude = np.random.uniform(0.5, 1.5)
        num_cycles = np.random.uniform(1.0, 5.0) # Number of full cycles in the signal
        frequency = num_cycles * 2 * np.pi
        phase = np.random.uniform(0, 2 * np.pi)
        # Generate clean signal
        clean_signal = amplitude * np.sin(frequency * time + phase)
        # Generate noise
        noise = np.random.normal(0, noise_std_dev, signal_length)
        # Generate noisy signal
        noisy_signal = clean_signal + noise
        clean_signals_list.append(clean_signal)
        noisy_signals_list.append(noisy_signal)
    # Convert lists to numpy arrays first for efficiency
    clean_signals_np = np.array(clean_signals_list, dtype=np.float32)
    noisy_signals_np = np.array(noisy_signals_list, dtype=np.float32)
    # Convert numpy arrays to torch tensors and add channel dimension
    # Shape: [n_samples, 1, signal_length] - Adding channel dim for Conv1d compatibility later
    clean_signals = torch.from_numpy(clean_signals_np).unsqueeze(1)
    noisy_signals = torch.from_numpy(noisy_signals_np).unsqueeze(1)
    return clean_signals, noisy_signals

# Check if datasets already exist, if not generate and save them
train_clean_path = train_dir / "clean_signals.pt"
test_clean_path = test_dir / "clean_signals.pt"

if not train_clean_path.exists() or not test_clean_path.exists():
    # Generate Training and Test Data
    clean_train_data, noisy_train_data = generate_sine_wave_dataset(
        N_SAMPLES_TRAIN, SIGNAL_LENGTH, NOISE_STD_DEV
    )
    clean_test_data, noisy_test_data = generate_sine_wave_dataset(
        N_SAMPLES_TEST, SIGNAL_LENGTH, NOISE_STD_DEV
    )
    
    # Save datasets
    torch.save(clean_train_data, train_dir / "clean_signals.pt")
    torch.save(noisy_train_data, train_dir / "noisy_signals.pt")
    torch.save(clean_test_data, test_dir / "clean_signals.pt")
    torch.save(noisy_test_data, test_dir / "noisy_signals.pt")
    
    print(f"Training data shape: Clean={clean_train_data.shape}, Noisy={noisy_train_data.shape}")
    print(f"Test data shape: Clean={clean_test_data.shape}, Noisy={noisy_test_data.shape}")
else:
    # Load existing datasets
    clean_train_data = torch.load(train_dir / "clean_signals.pt")
    noisy_train_data = torch.load(train_dir / "noisy_signals.pt")
    clean_test_data = torch.load(test_dir / "clean_signals.pt")
    noisy_test_data = torch.load(test_dir / "noisy_signals.pt")
    
    print(f"Loaded existing datasets.")
    print(f"Training data shape: Clean={clean_train_data.shape}, Noisy={noisy_train_data.shape}")
    print(f"Test data shape: Clean={clean_test_data.shape}, Noisy={noisy_test_data.shape}")

# # --- Visualize an Example ---
# sample_idx = 0
# plt.figure(figsize=(12, 4))
# plt.plot(clean_train_data[sample_idx, 0, :].numpy(), label='Clean Signal', alpha=0.8)
# plt.plot(noisy_train_data[sample_idx, 0, :].numpy(), label=f'Noisy Signal (std={NOISE_STD_DEV})', alpha=0.7)
# plt.title(f"Example Signal Sample #{sample_idx}")
# plt.xlabel("Time Step")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)
# plt.show()

