"""
Sharp World Model with:
1. VQ-VAE style vector quantization for discrete latents
2. Discriminator for adversarial sharpness
3. Skip connections (U-Net style) for detail preservation
4. Perceptual loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.datasets import MovingMNISTDataset
from utils import get_device


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for discrete latents."""
    
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, x):
        # x: (B, C, H, W) where C = embedding_dim
        B, C, H, W = x.shape
        
        # Flatten to (B*H*W, C)
        flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Calculate distances to embeddings
        distances = (flat.pow(2).sum(1, keepdim=True) 
                    + self.embedding.weight.pow(2).sum(1)
                    - 2 * flat @ self.embedding.weight.t())
        
        # Get nearest embedding indices
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, vq_loss, indices.view(B, H, W)


class ResBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class UNetEncoder(nn.Module):
    """U-Net style encoder with skip connections."""
    
    def __init__(self, in_channels=3, base_channels=64, latent_channels=64):
        super().__init__()
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2),
            ResBlock(base_channels),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2),
            ResBlock(base_channels*2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2),
            ResBlock(base_channels*4),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels*4, latent_channels, 4, 2, 1),  # 8 -> 4
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)   # 32x32
        e2 = self.enc2(e1)  # 16x16
        e3 = self.enc3(e2)  # 8x8
        e4 = self.enc4(e3)  # 4x4
        return e4, [e1, e2, e3]


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections."""
    
    def __init__(self, out_channels=3, base_channels=64, latent_channels=64):
        super().__init__()
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels*4, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4*2, base_channels*2, 4, 2, 1),  # 8 -> 16 (with skip)
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            ResBlock(base_channels*2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2*2, base_channels, 4, 2, 1),  # 16 -> 32 (with skip)
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResBlock(base_channels),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1),  # 32 -> 64 (with skip)
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, z, skips):
        e1, e2, e3 = skips
        
        d4 = self.dec4(z)                          # 8x8
        d3 = self.dec3(torch.cat([d4, e3], dim=1)) # 16x16
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # 32x32
        d1 = self.dec1(torch.cat([d2, e1], dim=1)) # 64x64
        
        return d1


class Discriminator(nn.Module):
    """PatchGAN discriminator for adversarial training."""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, 1, 0),  # Output: (B, 1, 1, 1)
        )
    
    def forward(self, x):
        return self.model(x).view(-1)


class SharpWorldModel(nn.Module):
    """World model with VQ-VAE and skip connections for sharp outputs."""
    
    def __init__(self, latent_channels=64, num_embeddings=512):
        super().__init__()
        
        self.encoder = UNetEncoder(in_channels=3, base_channels=64, latent_channels=latent_channels)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_channels)
        self.decoder = UNetDecoder(out_channels=3, base_channels=64, latent_channels=latent_channels)
        
        # GRU for dynamics in latent space (4x4 spatial)
        self.latent_dim = latent_channels * 4 * 4  # 64 * 16 = 1024
        self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=2, batch_first=True)
        self.latent_pred = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
    
    def encode(self, x):
        """Encode image to quantized latent."""
        z, skips = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        return z_q, skips, vq_loss
    
    def decode(self, z, skips):
        """Decode quantized latent to image."""
        return self.decoder(z, skips)
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W)
        """
        B, T, C, H, W = frames.shape
        
        # Encode all frames
        all_z = []
        all_skips = []
        total_vq_loss = 0
        
        for t in range(T):
            z_q, skips, vq_loss = self.encode(frames[:, t])
            all_z.append(z_q)
            all_skips.append(skips)
            total_vq_loss += vq_loss
        
        total_vq_loss /= T
        
        # Stack latents: (B, T, C, 4, 4) -> (B, T, C*16)
        z_stack = torch.stack(all_z, dim=1)  # (B, T, C, 4, 4)
        z_flat = z_stack.view(B, T, -1)  # (B, T, C*16)
        
        # Reconstruct all frames
        recon = []
        for t in range(T):
            recon.append(self.decode(all_z[t], all_skips[t]))
        recon = torch.stack(recon, dim=1)
        
        # Predict next frames
        pred = []
        gru_out, _ = self.gru(z_flat[:, :-1])  # (B, T-1, latent_dim)
        
        for t in range(T - 1):
            z_pred_flat = self.latent_pred(gru_out[:, t])  # (B, latent_dim)
            z_pred = z_pred_flat.view(B, -1, 4, 4)  # (B, C, 4, 4)
            # Use current frame's skip connections for prediction
            pred.append(self.decode(z_pred, all_skips[t]))
        
        pred = torch.stack(pred, dim=1)
        
        return {
            'recon': recon,
            'pred': pred,
            'vq_loss': total_vq_loss,
            'z': z_stack,
        }
    
    def autoregressive_rollout(self, initial_frames, num_steps):
        """Generate future frames."""
        B, T, C, H, W = initial_frames.shape
        
        # Encode initial frames
        all_z = []
        last_skips = None
        
        for t in range(T):
            z_q, skips, _ = self.encode(initial_frames[:, t])
            all_z.append(z_q)
            last_skips = skips
        
        z_history = torch.stack(all_z, dim=1).view(B, T, -1)
        
        generated = []
        hidden = None
        
        for step in range(num_steps):
            # GRU prediction
            gru_out, hidden = self.gru(z_history, hidden)
            z_pred_flat = self.latent_pred(gru_out[:, -1])
            z_pred = z_pred_flat.view(B, -1, 4, 4)
            
            # Decode
            frame = self.decode(z_pred, last_skips)
            generated.append(frame)
            
            # Update history
            z_history = torch.cat([z_history[:, 1:], z_pred_flat.unsqueeze(1)], dim=1)
        
        return torch.stack(generated, dim=1)


def train():
    device = get_device()
    print(f"Device: {device}")
    
    # Data
    dataset = MovingMNISTDataset("data/moving_mnist.npy", seq_length=10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Models
    model = SharpWorldModel(latent_channels=64, num_embeddings=512).to(device)
    discriminator = Discriminator().to(device)
    
    params_g = sum(p.numel() for p in model.parameters())
    params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params: {params_g:,}")
    print(f"Discriminator params: {params_d:,}")
    print(f"Total: {params_g + params_d:,}")
    
    # Optimizers
    opt_g = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Training
    print("\n" + "="*60)
    print("Training Sharp World Model (VQ-VAE + GAN)")
    print("="*60)
    
    best_loss = float('inf')
    ckpt_path = 'checkpoints/sharp_world_model.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(20):
        model.train()
        discriminator.train()
        
        epoch_loss_g = 0
        epoch_loss_d = 0
        epoch_recon = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            
            # ===== Train Discriminator =====
            opt_d.zero_grad()
            
            with torch.no_grad():
                out = model(frames)
            
            real_frames = frames.view(-1, C, H, W)
            fake_frames = out['recon'].view(-1, C, H, W).detach()
            
            # Real/fake labels
            real_labels = torch.ones(B * T, device=device) * 0.9  # Label smoothing
            fake_labels = torch.zeros(B * T, device=device) + 0.1
            
            # Discriminator loss
            d_real = discriminator(real_frames)
            d_fake = discriminator(fake_frames)
            
            loss_d_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            loss_d.backward()
            opt_d.step()
            
            # ===== Train Generator =====
            opt_g.zero_grad()
            
            out = model(frames)
            
            # Reconstruction loss
            loss_recon = F.mse_loss(out['recon'], frames)
            
            # Prediction loss
            loss_pred = F.mse_loss(out['pred'], frames[:, 1:])
            
            # VQ loss
            loss_vq = out['vq_loss']
            
            # Adversarial loss (fool discriminator)
            fake_frames = out['recon'].view(-1, C, H, W)
            d_fake = discriminator(fake_frames)
            loss_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            
            # Total generator loss
            loss_g = loss_recon + loss_pred + 0.1 * loss_vq + 0.01 * loss_adv
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()
            
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            epoch_recon += loss_recon.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | "
                      f"G: {loss_g.item():.4f} | D: {loss_d.item():.4f} | "
                      f"Recon: {loss_recon.item():.4f}")
        
        n = len(loader)
        elapsed = time.time() - start
        avg_loss = epoch_loss_g / n
        
        print(f"Epoch {epoch+1}/20 | G: {avg_loss:.4f} | D: {epoch_loss_d/n:.4f} | "
              f"Recon: {epoch_recon/n:.4f} | Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'discriminator': discriminator.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  ✓ Saved best model")
    
    # Visualize
    print("\nGenerating visualization...")
    model.eval()
    
    with torch.no_grad():
        sample = next(iter(loader))[:4].to(device)
        out = model(sample)
        
        # Autoregressive
        initial = sample[:1, :5]
        generated = model.autoregressive_rollout(initial, num_steps=15)
    
    fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    
    # Row 1: Input
    for t in range(10):
        axes[0, t].imshow(sample[0, t].cpu().numpy().transpose(1, 2, 0))
        axes[0, t].axis('off')
        axes[0, t].set_title(f't={t}')
    
    # Row 2: Reconstruction
    for t in range(10):
        axes[1, t].imshow(out['recon'][0, t].cpu().numpy().transpose(1, 2, 0))
        axes[1, t].axis('off')
    
    # Row 3: Prediction
    for t in range(9):
        axes[2, t].imshow(out['pred'][0, t].cpu().numpy().transpose(1, 2, 0))
        axes[2, t].axis('off')
    axes[2, 9].axis('off')
    
    # Row 4-5: Autoregressive rollout
    for t in range(10):
        axes[3, t].imshow(generated[0, t].cpu().numpy().transpose(1, 2, 0))
        axes[3, t].axis('off')
        if t < 5:
            axes[3, t].set_title('seed' if t < 5 else '')
    
    for t in range(5):
        axes[4, t].imshow(generated[0, t+10].cpu().numpy().transpose(1, 2, 0))
        axes[4, t].axis('off')
    for t in range(5, 10):
        axes[4, t].axis('off')
    
    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Recon', fontsize=10)
    axes[2, 0].set_ylabel('Pred t+1', fontsize=10)
    axes[3, 0].set_ylabel('AR 0-9', fontsize=10)
    axes[4, 0].set_ylabel('AR 10-14', fontsize=10)
    
    plt.suptitle('Sharp World Model: VQ-VAE + GAN', fontsize=14)
    plt.tight_layout()
    plt.savefig('sharp_model_result.png', dpi=150)
    print("Saved sharp_model_result.png")
    
    print(f"\n✅ Training Complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
