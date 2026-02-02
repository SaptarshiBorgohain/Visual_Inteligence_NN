"""
Improved Sharp World Model v2
- 8x8x128 latent (instead of 4x4x64)
- ConvGRU for spatial dynamics
- Deeper decoder with residual blocks
- L1 + Perceptual loss
- Phased training
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


# ============================================================
# VECTOR QUANTIZER with EMA updates
# ============================================================
class VectorQuantizer(nn.Module):
    """Simple VQ layer without EMA (MPS compatible)."""
    
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0/num_embeddings, 1.0/num_embeddings)
        
    def forward(self, x):
        # x: (B, C, H, W) where C = embedding_dim
        B, C, H, W = x.shape
        
        # Flatten to (B*H*W, C)
        flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Calculate distances
        distances = (flat.pow(2).sum(1, keepdim=True) 
                    + self.embedding.weight.pow(2).sum(1)
                    - 2 * torch.matmul(flat, self.embedding.weight.t()))
        
        # Get nearest embedding indices
        indices = distances.argmin(dim=1)
        
        # Quantize
        quantized = self.embedding(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Codebook usage (for logging)
        unique_indices = torch.unique(indices)
        usage = len(unique_indices) / self.num_embeddings
        
        return quantized, vq_loss, indices.view(B, H, W), usage


# ============================================================
# RESIDUAL BLOCK
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2)


# ============================================================
# ENCODER - outputs 8x8x192 (~9M total)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=192):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(96, 192, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(192, latent_channels, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        e1 = self.enc1(x)   # 32x32x96
        e2 = self.enc2(e1)  # 16x16x192
        e3 = self.enc3(e2)  # 8x8x192
        return e3, [e1, e2]


# ============================================================
# DECODER - balanced channels
# ============================================================
class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_channels=192):
        super().__init__()
        
        # 8x8 -> 16x16
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 192, 4, 2, 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
        )
        
        # 16x16 -> 32x32 (with skip from enc2: 192+192=384)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(384, 96, 4, 2, 1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
        )
        
        # 32x32 -> 64x64 (with skip from enc1: 96+96=192)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, out_channels, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, z, skips):
        e1, e2 = skips
        
        d3 = self.dec3(z)                          # 16x16x128
        d2 = self.dec2(torch.cat([d3, e2], dim=1)) # 32x32x64
        d1 = self.dec1(torch.cat([d2, e1], dim=1)) # 64x64x3
        
        return d1


# ============================================================
# CONV GRU - Spatial dynamics
# ============================================================
class ConvGRUCell(nn.Module):
    """Convolutional GRU cell operating on spatial feature maps."""
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        self.reset_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        
        self.hidden_channels = hidden_channels
    
    def forward(self, x, h):
        # x: (B, C, H, W), h: (B, hidden, H, W)
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), 
                           device=x.device, dtype=x.dtype)
        
        combined = torch.cat([x, h], dim=1)
        
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        
        combined_reset = torch.cat([x, r * h], dim=1)
        candidate = torch.tanh(self.candidate(combined_reset))
        
        h_new = (1 - z) * h + z * candidate
        return h_new


class ConvGRU(nn.Module):
    """Multi-layer ConvGRU for spatial dynamics prediction."""
    
    def __init__(self, channels=192, hidden_channels=192, num_layers=2, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            ConvGRUCell(channels if i == 0 else hidden_channels, hidden_channels, kernel_size)
            for i in range(num_layers)
        ])
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, 3, 1, 1),
        )
    
    def forward(self, x_seq, hidden=None):
        """
        Args:
            x_seq: (B, T, C, H, W)
            hidden: list of hidden states per layer
        Returns:
            outputs: (B, T, C, H, W)
            hidden: updated hidden states
        """
        B, T, C, H, W = x_seq.shape
        
        if hidden is None:
            hidden = [None] * self.num_layers
        
        outputs = []
        
        for t in range(T):
            x = x_seq[:, t]
            
            for i, cell in enumerate(self.cells):
                x = cell(x, hidden[i])
                hidden[i] = x
            
            out = self.output_conv(x)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1), hidden
    
    def predict_next(self, z, hidden=None):
        """Predict single next latent."""
        if hidden is None:
            hidden = [None] * self.num_layers
        
        x = z
        for i, cell in enumerate(self.cells):
            x = cell(x, hidden[i])
            hidden[i] = x
        
        return self.output_conv(x), hidden


# ============================================================
# PERCEPTUAL LOSS (Simple VGG-like)
# ============================================================
class PerceptualLoss(nn.Module):
    """Simple perceptual loss using conv features."""
    
    def __init__(self):
        super().__init__()
        # Simple feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        
        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        return F.l1_loss(pred_feat, target_feat)


# ============================================================
# DISCRIMINATOR - PatchGAN
# ============================================================
class Discriminator(nn.Module):
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
            
            nn.Conv2d(512, 1, 4, 1, 0),
        )
    
    def forward(self, x):
        return self.model(x).view(-1)


# ============================================================
# SHARP WORLD MODEL V2
# ============================================================
class SharpWorldModelV2(nn.Module):
    """Improved world model with 8x8x192 latent and ConvGRU. ~9M params."""
    
    def __init__(self, latent_channels=192, num_embeddings=512):
        super().__init__()
        
        self.encoder = Encoder(in_channels=3, latent_channels=latent_channels)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, 
                                  embedding_dim=latent_channels,
                                  commitment_cost=0.25)
        self.decoder = Decoder(out_channels=3, latent_channels=latent_channels)
        self.dynamics = ConvGRU(channels=latent_channels, hidden_channels=latent_channels, 
                                num_layers=2, kernel_size=3)
    
    def encode(self, x):
        z, skips = self.encoder(x)
        z_q, vq_loss, indices, usage = self.vq(z)
        return z_q, skips, vq_loss, usage
    
    def decode(self, z, skips):
        return self.decoder(z, skips)
    
    def forward(self, frames, use_dynamics=False):
        """
        Args:
            frames: (B, T, C, H, W)
            use_dynamics: whether to use GRU for prediction
        """
        B, T, C, H, W = frames.shape
        
        # Encode all frames
        all_z = []
        all_skips = []
        total_vq_loss = 0
        total_usage = 0
        
        for t in range(T):
            z_q, skips, vq_loss, usage = self.encode(frames[:, t])
            all_z.append(z_q)
            all_skips.append(skips)
            total_vq_loss += vq_loss
            total_usage += usage
        
        total_vq_loss /= T
        avg_usage = total_usage / T
        
        # Reconstruct all frames
        recon = []
        for t in range(T):
            recon.append(self.decode(all_z[t], all_skips[t]))
        recon = torch.stack(recon, dim=1)
        
        result = {
            'recon': recon,
            'vq_loss': total_vq_loss,
            'codebook_usage': avg_usage,
            'z': torch.stack(all_z, dim=1),  # (B, T, C, 8, 8)
        }
        
        # Dynamics prediction
        if use_dynamics:
            z_stack = torch.stack(all_z, dim=1)  # (B, T, C, H, W)
            z_pred, _ = self.dynamics(z_stack[:, :-1])  # Predict from t to t+1
            
            # Decode predictions
            pred = []
            for t in range(T - 1):
                pred.append(self.decode(z_pred[:, t], all_skips[t]))
            
            result['pred'] = torch.stack(pred, dim=1)
            result['z_pred'] = z_pred
        
        return result
    
    def autoregressive_rollout(self, initial_frames, num_steps):
        """Generate future frames autoregressively."""
        B, T, C, H, W = initial_frames.shape
        
        # Encode initial frames
        all_z = []
        last_skips = None
        
        for t in range(T):
            z_q, skips, _, _ = self.encode(initial_frames[:, t])
            all_z.append(z_q)
            last_skips = skips
        
        # Initialize dynamics with seed sequence
        z_stack = torch.stack(all_z, dim=1)
        _, hidden = self.dynamics(z_stack)
        
        generated = []
        z_current = all_z[-1]
        
        for step in range(num_steps):
            z_next, hidden = self.dynamics.predict_next(z_current, hidden)
            frame = self.decode(z_next, last_skips)
            generated.append(frame)
            z_current = z_next
        
        return torch.stack(generated, dim=1)


# ============================================================
# TRAINING
# ============================================================
def train():
    device = get_device()
    print(f"Device: {device}")
    
    # Data - shorter sequences for faster training
    dataset = MovingMNISTDataset("data/moving_mnist.npy", seq_length=5)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Models
    model = SharpWorldModelV2(latent_channels=192, num_embeddings=512).to(device)
    discriminator = Discriminator().to(device)
    perceptual = PerceptualLoss().to(device)
    
    params_g = sum(p.numel() for p in model.parameters())
    params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params: {params_g:,}")
    print(f"Discriminator params: {params_d:,}")
    print(f"Total: {params_g + params_d:,}")
    
    # Optimizers
    opt_g = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/sharp_world_model_v2.pt'
    
    # ========================================
    # PHASE 1: Reconstruction Only (3 epochs)
    # ========================================
    print("\n" + "="*60)
    print("PHASE 1: Reconstruction Only (3 epochs)")
    print("GAN OFF, Dynamics OFF")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        epoch_l1 = 0
        epoch_vq = 0
        epoch_usage = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            
            opt_g.zero_grad()
            
            out = model(frames, use_dynamics=False)
            
            # L1 reconstruction loss
            loss_l1 = F.l1_loss(out['recon'], frames)
            
            # VQ loss
            loss_vq = out['vq_loss']
            
            # Total (no perceptual in phase 1 for speed)
            loss = 1.0 * loss_l1 + 1.0 * loss_vq
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()
            
            # MPS sync
            if device.type == 'mps':
                torch.mps.synchronize()
            
            epoch_loss += loss.item()
            epoch_l1 += loss_l1.item()
            epoch_vq += loss_vq.item()
            epoch_usage += out['codebook_usage']
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | L1: {loss_l1.item():.4f} | "
                      f"VQ: {loss_vq.item():.4f} | Usage: {out['codebook_usage']*100:.1f}%")
        
        n = len(loader)
        elapsed = time.time() - start
        avg_loss = epoch_loss / n
        avg_usage = epoch_usage / n
        
        print(f"Epoch {epoch+1}/3 | Loss: {avg_loss:.4f} | L1: {epoch_l1/n:.4f} | "
              f"VQ: {epoch_vq/n:.4f} | Usage: {avg_usage*100:.1f}% | Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'phase': 1, 'model': model.state_dict(), 
                       'loss': avg_loss}, ckpt_path)
            print(f"  ✓ Saved")
        
        # Early stopping check
        if avg_usage > 0.5:
            print(f"  ✓ Codebook usage > 50%, good convergence")
    
    # ========================================
    # PHASE 2: Add GAN (3 epochs)
    # ========================================
    print("\n" + "="*60)
    print("PHASE 2: Add GAN (3 epochs)")
    print("Freeze encoder, train decoder + discriminator")
    print("="*60)
    
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.vq.parameters():
        param.requires_grad = False
    
    for epoch in range(3):
        model.train()
        discriminator.train()
        epoch_loss_g = 0
        epoch_loss_d = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            
            # ===== Train Discriminator =====
            opt_d.zero_grad()
            
            with torch.no_grad():
                out = model(frames, use_dynamics=False)
            
            real = frames.view(-1, C, H, W)
            fake = out['recon'].view(-1, C, H, W).detach()
            
            d_real = discriminator(real)
            d_fake = discriminator(fake)
            
            loss_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real) * 0.9) +
                     F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake) + 0.1)) / 2
            
            loss_d.backward()
            opt_d.step()
            
            # ===== Train Generator (decoder only) =====
            opt_g.zero_grad()
            
            out = model(frames, use_dynamics=False)
            
            # L1 + Perceptual
            recon_flat = out['recon'].view(-1, C, H, W)
            frames_flat = frames.view(-1, C, H, W)
            
            loss_l1 = F.l1_loss(recon_flat, frames_flat)
            loss_percep = perceptual(recon_flat, frames_flat)
            
            # GAN loss
            d_fake = discriminator(recon_flat)
            loss_gan = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            
            loss_g = 1.0 * loss_l1 + 0.1 * loss_percep + 0.05 * loss_gan
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()
            
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | G: {loss_g.item():.4f} | "
                      f"D: {loss_d.item():.4f} | L1: {loss_l1.item():.4f}")
        
        n = len(loader)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1}/3 | G: {epoch_loss_g/n:.4f} | D: {epoch_loss_d/n:.4f} | Time: {elapsed:.1f}s")
        
        torch.save({'epoch': epoch, 'phase': 2, 'model': model.state_dict(),
                   'discriminator': discriminator.state_dict()}, ckpt_path)
        print(f"  ✓ Saved")
    
    # ========================================
    # PHASE 3: Add Dynamics (2 epochs)
    # ========================================
    print("\n" + "="*60)
    print("PHASE 3: Add Dynamics (2 epochs)")
    print("Train ConvGRU to predict next latent")
    print("="*60)
    
    # Unfreeze encoder, freeze decoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.vq.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    for param in model.dynamics.parameters():
        param.requires_grad = True
    
    opt_dyn = optim.AdamW(model.dynamics.parameters(), lr=1e-4)
    
    for epoch in range(2):
        model.train()
        epoch_loss = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            
            opt_dyn.zero_grad()
            
            out = model(frames, use_dynamics=True)
            
            # Latent prediction loss
            z_next_gt = out['z'][:, 1:]  # Ground truth next latents
            loss_latent = F.mse_loss(out['z_pred'], z_next_gt)
            
            # Frame prediction loss
            frames_next = frames[:, 1:]
            loss_pred = F.l1_loss(out['pred'], frames_next)
            
            loss = loss_latent + 0.5 * loss_pred
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 1.0)
            opt_dyn.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | Latent: {loss_latent.item():.4f} | "
                      f"Pred: {loss_pred.item():.4f}")
        
        n = len(loader)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch+1}/2 | Loss: {epoch_loss/n:.4f} | Time: {elapsed:.1f}s")
        
        torch.save({'epoch': epoch, 'phase': 3, 'model': model.state_dict(),
                   'discriminator': discriminator.state_dict()}, ckpt_path)
        print(f"  ✓ Saved")
    
    # ========================================
    # VISUALIZATION
    # ========================================
    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)
    
    # Unfreeze all for eval
    for param in model.parameters():
        param.requires_grad = True
    
    model.eval()
    
    with torch.no_grad():
        sample = next(iter(loader))[:4].to(device)
        T = sample.shape[1]  # Actual sequence length
        out = model(sample, use_dynamics=True)
        
        # Autoregressive rollout
        initial = sample[:1, :3]
        generated = model.autoregressive_rollout(initial, num_steps=10)
    
    fig, axes = plt.subplots(4, T, figsize=(T*2, 8))
    
    # Row 1: Input
    for t in range(T):
        axes[0, t].imshow(sample[0, t].cpu().numpy().transpose(1, 2, 0))
        axes[0, t].axis('off')
        axes[0, t].set_title(f't={t}')
    
    # Row 2: Reconstruction
    for t in range(T):
        axes[1, t].imshow(out['recon'][0, t].cpu().numpy().transpose(1, 2, 0))
        axes[1, t].axis('off')
    
    # Row 3: Prediction (one less than T)
    for t in range(min(T-1, out['pred'].shape[1])):
        axes[2, t].imshow(out['pred'][0, t].cpu().numpy().transpose(1, 2, 0))
        axes[2, t].axis('off')
    if T > out['pred'].shape[1]:
        axes[2, T-1].axis('off')
    
    # Row 4: Autoregressive
    for t in range(min(T, generated.shape[1])):
        axes[3, t].imshow(generated[0, t].cpu().numpy().transpose(1, 2, 0))
        axes[3, t].axis('off')
    
    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Recon', fontsize=10)
    axes[2, 0].set_ylabel('Pred t+1', fontsize=10)
    axes[3, 0].set_ylabel('AR', fontsize=10)
    
    plt.suptitle('Sharp World Model V2 (VQ-VAE + ConvGRU)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/sharp_model_v2_result.png', dpi=150)
    print("Saved results/sharp_model_v2_result.png")
    
    print("\n✅ Training Complete!")
    print(f"   Model saved to: {ckpt_path}")


if __name__ == "__main__":
    train()
