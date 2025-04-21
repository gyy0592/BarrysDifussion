import math
import torch
import torch.nn.functional as F
import torch.nn as nn

# Cell 3: DDPM Network Architecture for 1D Signals - Configurable Version

class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings using sinusoidal position encoding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """Basic convolutional block with normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=None, model_config=None):
        super().__init__()
        self.model_config = model_config
        kernel_size = kernel_size or self.model_config["kernel_size"]
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size//2
        )
        self.norm = nn.GroupNorm(
            num_groups=min(self.model_config["min_groups"], out_channels), 
            num_channels=out_channels
        )
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block with time embedding injection"""
    def __init__(self, in_channels, out_channels, time_channels, kernel_size=None, model_config=None):
        super().__init__()
        self.model_config = model_config
        kernel_size = kernel_size or self.model_config["kernel_size"]
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # First convolution block
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, self.model_config)
        
        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(self.model_config["min_groups"], out_channels), 
                num_channels=out_channels
            ),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        )
        
        # Residual connection if input and output channels differ
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, time_emb):
        # Residual branch
        residual = self.residual_conv(x)
        
        # Main branch
        h = self.conv1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)[:, :, None]  # [B, C, 1]
        h = h + time_emb
        
        # Second convolution
        h = self.conv2(h)
        
        # Add residual connection
        return h + residual

class Attention1D(nn.Module):
    """Self-attention layer for 1D signals"""
    def __init__(self, channels, model_config=None):
        super().__init__()
        self.model_config = model_config
        self.norm = nn.GroupNorm(
            num_groups=min(self.model_config["min_groups"], channels), 
            num_channels=channels
        )
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.k = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, kernel_size=1)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5
        
    def forward(self, x):
        B, C, L = x.shape
        
        # Normalize input
        h = self.norm(x)
        
        # Get q, k, v - using separate projections to avoid dimension issues
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Compute attention
        attn = torch.einsum('bcl,bcm->blm', q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        
        # Apply attention to values
        out = torch.einsum('blm,bcm->bcl', attn, v)
        
        # Final projection
        out = self.out(out)
        
        # Residual connection
        return out + x

class DDPM1D(nn.Module):
    """1D Denoising Diffusion Probabilistic Model with configurable dimensions"""
    def __init__(self, config):
        super().__init__()
        
        # Extract model config if full config is provided
        if isinstance(config, dict) and 'model' in config:
            config = config['model']
        
        # Convert channel_mults to tuple if it's a list
        if 'channel_mults' in config and isinstance(config['channel_mults'], list):
            config['channel_mults'] = tuple(config['channel_mults'])
            
        # Store the config
        self.model_config = config
        
        # Set model parameters directly from config
        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.time_dim = config["time_dim"]
        self.base_channels = config["base_channels"]
        self.channel_mults = config["channel_mults"]
        self.use_attention = config["use_attention"]
        self.direct_path_scale = config["direct_path_scale"]
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_dim // 4),
            nn.Linear(self.time_dim // 4, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        # Initial projection from input to feature space
        self.init_conv = nn.Conv1d(self.in_channels, self.base_channels, 
                                  kernel_size=self.model_config["kernel_size"], 
                                  padding=self.model_config["kernel_size"]//2)
        
        # Direct path for better signal preservation
        self.direct = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
        
        # Create channel dimensions for each level
        self.channels = [self.base_channels]
        for mult in self.channel_mults:
            self.channels.append(self.base_channels * mult)
        
        # Downsampling
        self.downs = nn.ModuleList()
        input_channels = self.base_channels
        for i in range(len(self.channel_mults)):
            output_channels = self.channels[i+1]
            is_last = i == len(self.channel_mults) - 1
            self.downs.append(
                nn.ModuleList([
                    # Residual blocks
                    ResidualBlock(input_channels, output_channels, self.time_dim, model_config=self.model_config),
                    ResidualBlock(output_channels, output_channels, self.time_dim, model_config=self.model_config),
                    # Downsample (except for last layer)
                    nn.Conv1d(output_channels, output_channels, 4, 2, 1) if not is_last else nn.Identity()
                ])
            )
            input_channels = output_channels
        
        # Middle with attention
        mid_channels = self.channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, self.time_dim, model_config=self.model_config)
        self.mid_attn = Attention1D(mid_channels, model_config=self.model_config) if self.use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, self.time_dim, model_config=self.model_config)
        
        # Upsampling
        self.ups = nn.ModuleList()
        
        for i in reversed(range(len(self.channel_mults))):
            output_channels = self.channels[i]
            
            # Calculate skip connection channels
            skip_channels = self.channels[i+1]
            
            # Calculate the total channels after concatenation
            concat_channels = output_channels + skip_channels
            
            self.ups.append(
                nn.ModuleList([
                    # Upsample - from channels[i+1] to output_channels
                    nn.ConvTranspose1d(
                        self.channels[i+1], 
                        output_channels, 
                        kernel_size=4, stride=2, padding=1
                    ),
                    # First residual block takes concatenated input (concat_channels)
                    ResidualBlock(concat_channels, output_channels, self.time_dim, model_config=self.model_config),
                    # Second residual block
                    ResidualBlock(output_channels, output_channels, self.time_dim, model_config=self.model_config)
                ])
            )
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.GroupNorm(
                num_groups=min(self.model_config["min_groups"], self.base_channels), 
                num_channels=self.base_channels
            ),
            nn.SiLU(),
            nn.Conv1d(self.base_channels, self.out_channels, 
                     kernel_size=self.model_config["kernel_size"], 
                     padding=self.model_config["kernel_size"]//2)
        )
    
    def forward(self, x, time):
        # Direct path (signal preservation)
        direct = self.direct(x) * self.direct_path_scale
        
        # Time embedding
        t = self.time_mlp(time)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Store skip connections
        skips = [h]
        
        # Downsampling
        for down_block in self.downs:
            res1, res2, downsample = down_block
            
            h = res1(h, t)
            h = res2(h, t)
            skips.append(h)
            h = downsample(h)
        
        # Middle blocks
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)
        
        # Upsampling with skip connections
        for up_block in self.ups:
            upsample, res1, res2 = up_block
            
            # Upsample
            h = upsample(h)
            
            # Get skip connection
            skip = skips.pop()
            
            # Ensure dimensions match for concatenation
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
            
            # Concatenate along channel dimension
            h = torch.cat([h, skip], dim=1)
            
            # Apply residual blocks
            h = res1(h, t)
            h = res2(h, t)
        
        # Final output
        h = self.final_conv(h)
        
        # Add direct path for better signal preservation
        return h + direct