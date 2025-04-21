import os
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .diffusion_utils import p_losses
from .diffusion_const import calculate_diffusion_constants
import matplotlib.pyplot as plt
import csv  # Added CSV import


def train_diffusion_denoiser_1d(model, train_loader, test_loader, device, EPOCHS=None, TIMESTEPS=None, LEARNING_RATE=None, config = None):
    # Extract configurations
    general_config = config['general']
    diffusion_config = config['diffusion']  
    TIMESTEPS = diffusion_config['timesteps']
    diffusion_consts = calculate_diffusion_constants(TIMESTEPS, device)
    sqrt_alphas_cumprod = diffusion_consts['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_consts['sqrt_one_minus_alphas_cumprod']
    sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod
    # Set variables from config
    LEARNING_RATE = general_config['learning_rate']
    EPOCHS = general_config['epochs']
    # Initialize optimizer with AdamW
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/10)

    # Tracking variables
    train_losses_epoch, val_losses_epoch = [], []
    best_val_loss, best_epoch = float('inf'), -1
    
    
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
    save_path = "output/best_diffusion_denoiser_1d.pth"
    # Create CSV file for training info
    csv_path = "output/training_info.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Learning Rate', 
                             'Time Elapsed (s)', 'ETA (s)', 'Memory Usage (GB)'])

    print(f"[{time.strftime('%H:%M:%S')}] Starting training: {EPOCHS} epochs on {device}")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            batch_count += 1
            optimizer.zero_grad()
            
            x_start = batch[0].to(device)
            t = torch.randint(0, TIMESTEPS, (x_start.shape[0],), device=device).long()
            
            loss = p_losses(model, x_start, t, loss_type="l2", sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for val_batch in test_loader:
                x_val_start = val_batch[0].to(device)
                t_val = torch.randint(0, TIMESTEPS, (x_val_start.shape[0],), device=device).long()
                loss_val = p_losses(model, x_val_start, t_val, loss_type="l2", sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)
                epoch_val_loss += loss_val.item()
        
        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(test_loader)
        train_losses_epoch.append(avg_train_loss)
        val_losses_epoch.append(avg_val_loss)
        
        # Simple YOLO-style output format
        curr_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (EPOCHS - epoch - 1)
        
        # Get total memory usage in GB
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            mem_str = f"{mem:.2f}G"
            mem_val = f"{mem:.2f}"
        else:
            mem_str = "N/A"
            mem_val = "N/A"
        
        # Simple text output in YOLO-style format
        print(f"[{time.strftime('%H:%M:%S')}] {epoch+1}/{EPOCHS} t={avg_train_loss:.4f} v={avg_val_loss:.4f} lr={curr_lr:.6f} {elapsed:.1f}s ETA:{eta:.1f}s {mem_str}")
        
        # Write epoch info to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}", 
                                f"{curr_lr:.6f}", f"{elapsed:.1f}", f"{eta:.1f}", mem_val])
        
        # Mark best model with an asterisk
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"[{time.strftime('%H:%M:%S')}] * Best model saved: val_loss={best_val_loss:.4f}")
        
        # Early stopping
        if epoch > 20 and epoch - best_epoch >= 15:
            print(f"[{time.strftime('%H:%M:%S')}] Early stopping at epoch {epoch+1}")
            break

    print(f"[{time.strftime('%H:%M:%S')}] Done! Best: epoch {best_epoch}, val_loss={best_val_loss:.4f}, time={(time.time() - start_time)/60:.1f}min")
    print(f"[{time.strftime('%H:%M:%S')}] Training info saved to {csv_path}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses_epoch) + 1), train_losses_epoch, label='Training Loss', marker='o', linestyle='-', markersize=3)
    plt.plot(range(1, len(val_losses_epoch) + 1), val_losses_epoch, label='Validation Loss', marker='x', linestyle='--', markersize=3)
    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch"), plt.ylabel("MSE Loss")
    plt.grid(True), plt.legend()
    plt.savefig("output/loss_curves.png")