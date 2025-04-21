
import torch


def get_cosine_schedule(timesteps, s=0.008):
    """Creates a beta schedule that follows a cosine curve."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.02)

def calculate_diffusion_constants(timesteps, device=None):
    """Calculate all diffusion-related constants from timesteps."""
    betas = get_cosine_schedule(timesteps)
    
    # Pre-calculate diffusion constants
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    # Forward process constants
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # Reverse process constants
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    posterior_variance[0] = betas[0]
    
    # Additional sampling constant
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    constants = {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance,
        'sqrt_recip_alphas': sqrt_recip_alphas
    }
    
    # Move to device if specified
    if device:
        for k, v in constants.items():
            constants[k] = v.to(device)
            
    return constants