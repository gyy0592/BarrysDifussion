import torch
import torch.nn.functional as F
from tqdm import tqdm  # This imports the function
def extract(a, t, x_shape):
    """Extract coefficients at timesteps t and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, noise=None, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None):
    """Forward diffusion process: add noise to x_start according to schedule."""
    if noise is None:
        noise = torch.randn_like(x_start)
        
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2", sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None):
    """Calculate loss for training the diffusion model."""
    if noise is None:
        noise = torch.randn_like(x_start)
        
    # Forward process to get noisy signal x_t
    x_noisy, _ = q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    
    # Predict noise using model
    predicted_noise = denoise_model(x_noisy, t)
    
    # Loss calculation
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        # Huber loss (robust to outliers)
        loss = F.smooth_l1_loss(noise, predicted_noise)
        
    return loss



@torch.no_grad()
def p_sample(model, x, t, t_index, betas, alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    """
    Sample from p(x_{t-1} | x_t) - single step of the reverse process.
    """
    # Extract coefficients at timestep t
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = torch.sqrt(1.0 / extract(alphas, t, x.shape))
    
    # Predict noise component
    predicted_noise = model(x, t)
    
    # Calculate mean for the posterior
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    # Add noise except for the final step
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
                 
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def denoise_signal(model, initial_noisy_signal, start_step=None, betas=None, alphas=None, sqrt_one_minus_alphas_cumprod=None, posterior_variance=None, TIMESTEPS=None):
    """
    Perform the complete reverse diffusion process to denoise a signal.
    """
    model.eval()
    
    # Add batch dimension if needed
    if initial_noisy_signal.dim() == 2:
        initial_noisy_signal = initial_noisy_signal.unsqueeze(0)
    
    batch_size = initial_noisy_signal.shape[0]
    device = initial_noisy_signal.device
    
    # Initialize with pure noise or the given noisy signal
    x_t = initial_noisy_signal.to(device)
    
    # Determine starting timestep
    if start_step is None:
        start_step = TIMESTEPS - 1
    else:
        start_step = min(max(0, start_step), TIMESTEPS - 1)
    
    print(f"Starting denoising from step {start_step}...")
    
    # Incrementally denoise the signal
    denoising_progress = tqdm(reversed(range(0, start_step + 1)), desc="Denoising")
    for time_step in denoising_progress:
        # Get timestep tensor (same timestep for entire batch)
        t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
        
        # Sample from p(x_{t-1} | x_t)
        x_t = p_sample(model, x_t, t, time_step, betas, alphas, sqrt_one_minus_alphas_cumprod, posterior_variance)
    
    return x_t