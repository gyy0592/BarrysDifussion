import torch
import numpy as np
import matplotlib.pyplot as plt
from .diffusion_const import calculate_diffusion_constants
from .diffusion_utils import denoise_signal
import torch.nn.functional as F
import os
import csv
import time

def evaluate_model(model, test_loader, config, device):
    # Extract configurations
    general_config = config['general']
    diffusion_config = config['diffusion']

    # Set variables from config
    SIGNAL_LENGTH = general_config['signal_length']
    TIMESTEPS = diffusion_config['timesteps']

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate all diffusion constants and move to device
    diffusion_consts = calculate_diffusion_constants(TIMESTEPS, device)

    # Unpack constants for easier access if needed
    betas = diffusion_consts['betas']
    alphas = diffusion_consts['alphas']
    sqrt_one_minus_alphas_cumprod = diffusion_consts['sqrt_one_minus_alphas_cumprod']
    posterior_variance = diffusion_consts['posterior_variance']
    
    # Parameters
    N_EXAMPLES_TO_SHOW = 5
    denoising_start_step = TIMESTEPS - 1

    # Get test samples
    test_loader_iter = iter(test_loader)
    test_samples = next(test_loader_iter)
    clean_signals_batch = test_samples[0][:N_EXAMPLES_TO_SHOW].to(device)
    noisy_signals_batch = test_samples[1][:N_EXAMPLES_TO_SHOW].to(device)

    # Denoise the signals
    print("Starting denoising process for evaluation...")
    # Example modification to denoise_signal call in myeval.py
    estimated_step = calculate_equivalent_timestep(noisy_signals_batch, clean_signals_batch, diffusion_consts, TIMESTEPS)
    denoising_start_step = estimated_step
    denoised_signals_batch = denoise_signal(model, noisy_signals_batch, start_step=denoising_start_step, betas=betas, alphas=alphas, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, posterior_variance=posterior_variance, TIMESTEPS=TIMESTEPS)

    # Move to CPU for plotting
    clean_signals_cpu = clean_signals_batch.cpu().numpy()
    noisy_signals_cpu = noisy_signals_batch.cpu().numpy()
    denoised_signals_cpu = denoised_signals_batch.cpu().numpy()

    # Calculate MSE
    mse_noisy = F.mse_loss(noisy_signals_batch, clean_signals_batch).item()
    mse_denoised = F.mse_loss(denoised_signals_batch, clean_signals_batch).item()
    improvement = (mse_noisy - mse_denoised) / mse_noisy * 100 if mse_noisy > 0 else 0

    print(f"\nEvaluation Results (on {N_EXAMPLES_TO_SHOW} examples):")
    print(f"MSE (Noisy vs Clean): {mse_noisy:.6f}")
    print(f"MSE (Denoised vs Clean): {mse_denoised:.6f}")
    print(f"MSE Improvement: {improvement:.2f}%")
    
    # Create CSV file for evaluation metrics
    eval_csv_path = f"{output_dir}/evaluation_metrics.csv"
    
    # Write overall metrics
    with open(eval_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Metric', 'Value'])
        csv_writer.writerow(['MSE (Noisy vs Clean)', f"{mse_noisy:.6f}"])
        csv_writer.writerow(['MSE (Denoised vs Clean)', f"{mse_denoised:.6f}"])
        csv_writer.writerow(['MSE Improvement (%)', f"{improvement:.2f}"])
        csvfile.write('\n')
        
        # Write per-example metrics
        csv_writer.writerow(['Example', 'MSE (Noisy)', 'MSE (Denoised)', 'Improvement (%)'])
        
        # Plot results
        time_axis = np.arange(SIGNAL_LENGTH)
        example_metrics = []

        for i in range(N_EXAMPLES_TO_SHOW):
            # Calculate MSE for this example
            example_mse_noisy = F.mse_loss(noisy_signals_batch[i], clean_signals_batch[i]).item()
            example_mse_denoised = F.mse_loss(denoised_signals_batch[i], clean_signals_batch[i]).item()
            example_improvement = (example_mse_noisy - example_mse_denoised) / example_mse_noisy * 100 if example_mse_noisy > 0 else 0
            
            # Store metrics for CSV
            example_metrics.append([i+1, example_mse_noisy, example_mse_denoised, example_improvement])
            csv_writer.writerow([i+1, f"{example_mse_noisy:.6f}", f"{example_mse_denoised:.6f}", f"{example_improvement:.2f}"])
            
            # Plot signals
            plt.figure(figsize=(12, 4))
            plt.plot(time_axis, clean_signals_cpu[i, 0, :], 
                    label='Original Clean Signal', color='green', linewidth=2)
            plt.plot(time_axis, noisy_signals_cpu[i, 0, :], 
                    label=f'Noisy Signal (MSE: {example_mse_noisy:.4f})', color='red', alpha=0.6)
            plt.plot(time_axis, denoised_signals_cpu[i, 0, :], 
                    label=f'Denoised Signal (MSE: {example_mse_denoised:.4f})', 
                    color='blue', linestyle='--', linewidth=2)
            
            plt.title(f"Denoising Result - Example {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            
            # Save the figure instead of showing it
            # plt.savefig(f"{output_dir}/denoising_example_{i+1}.png")
            # plt.close()
    
    # Create a summary plot with all examples
    plt.figure(figsize=(15, 10))
    for i in range(N_EXAMPLES_TO_SHOW):
        plt.subplot(N_EXAMPLES_TO_SHOW, 1, i+1)
        plt.plot(time_axis, clean_signals_cpu[i, 0, :], label='Clean', color='green', linewidth=1.5)
        plt.plot(time_axis, noisy_signals_cpu[i, 0, :], label='Noisy', color='red', alpha=0.4, linewidth=1)
        plt.plot(time_axis, denoised_signals_cpu[i, 0, :], label='Denoised', color='blue', linestyle='--', linewidth=1.5)
        
        if i == 0:
            plt.legend(loc='upper right')
        
        plt.title(f"Example {i+1} (Improvement: {example_metrics[i][3]:.2f}%)")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/denoising_summary.png")
    plt.close()
    
    print(f"Evaluation results saved to {eval_csv_path}")
    print(f"Plots saved to {output_dir}/")
    
    return mse_noisy, mse_denoised, improvement

def calculate_equivalent_timestep(noisy_signals, clean_signals, diffusion_consts, timesteps):
    """
    Calculate which timestep in the diffusion process has noise level 
    most similar to the test data.
    
    Args:
        noisy_signals: The noisy signals from test data
        clean_signals: The clean signals from test data
        diffusion_consts: Dictionary of diffusion constants
        timesteps: Total number of timesteps in diffusion
    
    Returns:
        The estimated timestep to start denoising from
    """
    # Calculate average noise level in test data
    noise = noisy_signals - clean_signals
    test_noise_power = torch.mean(torch.pow(noise, 2)).item()
    
    # Get the diffusion schedule constants
    sqrt_one_minus_alphas_cumprod = diffusion_consts['sqrt_one_minus_alphas_cumprod']
    
    # Find the timestep with the closest noise level
    best_timestep = 0
    min_diff = float('inf')
    
    # Check various timesteps to find the best match
    for t in range(timesteps):
        # Noise power at timestep t
        t_noise_power = sqrt_one_minus_alphas_cumprod[t].item() ** 2
        diff = abs(t_noise_power - test_noise_power)
        
        if diff < min_diff:
            min_diff = diff
            best_timestep = t
    
    return best_timestep
