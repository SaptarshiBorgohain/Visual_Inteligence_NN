"""
Full Visual Physics World Model Training
- Longer training with perceptual sharpening
- Autoregressive rollout (predict 20+ frames)
- Physics evaluation (collision/bounce detection)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.datasets import MovingMNISTDataset
from utils import get_device


class EnhancedWorldModel(nn.Module):
    """Enhanced world model with larger capacity for sharper results."""
    
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Deeper Encoder: 64x64x3 -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),     # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),   # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, latent_dim),
        )
        
        # Deeper Decoder: latent -> 64x64x3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 64x64
            nn.Sigmoid(),
        )
        
        # Multi-layer GRU for better dynamics
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=2, batch_first=True)
        self.latent_pred = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def predict_next_latent(self, z_seq):
        """Predict next latent from sequence of latents."""
        # z_seq: (B, T, L)
        out, hidden = self.gru(z_seq)
        z_next = self.latent_pred(out[:, -1])  # Use last output
        return z_next, hidden
    
    def forward(self, frames, teacher_forcing=True):
        """
        Args:
            frames: (B, T, C, H, W)
            teacher_forcing: if True, use ground truth for dynamics
        """
        B, T, C, H, W = frames.shape
        
        # Encode all frames
        frames_flat = frames.view(B * T, C, H, W)
        z_flat = self.encode(frames_flat)
        z = z_flat.view(B, T, -1)  # (B, T, L)
        
        # Reconstruct all frames
        recon_flat = self.decode(z_flat)
        recon = recon_flat.view(B, T, C, H, W)
        
        # Predict next frames - process all at once with GRU
        z_input = z[:, :-1]  # (B, T-1, L) - all except last
        gru_out, _ = self.gru(z_input)  # (B, T-1, L)
        z_pred = self.latent_pred(gru_out)  # (B, T-1, L)
        
        # Decode predictions
        z_pred_flat = z_pred.reshape(B * (T-1), -1)
        frames_pred_flat = self.decode(z_pred_flat)
        frames_pred = frames_pred_flat.view(B, T-1, C, H, W)
        
        return {
            'z': z,
            'z_pred': z_pred,
            'recon': recon,
            'pred': frames_pred,
        }
    
    def autoregressive_rollout(self, initial_frames, num_steps):
        """
        Generate future frames autoregressively.
        Args:
            initial_frames: (B, T_init, C, H, W) - seed frames
            num_steps: number of future frames to generate
        Returns:
            generated: (B, num_steps, C, H, W)
        """
        B, T_init, C, H, W = initial_frames.shape
        
        # Encode initial frames
        frames_flat = initial_frames.view(B * T_init, C, H, W)
        z_flat = self.encode(frames_flat)
        z_history = z_flat.view(B, T_init, -1)  # (B, T_init, L)
        
        generated = []
        
        for step in range(num_steps):
            # Predict next latent from history
            z_next, _ = self.predict_next_latent(z_history)
            
            # Decode to image
            frame_next = self.decode(z_next)
            generated.append(frame_next)
            
            # Add to history
            z_history = torch.cat([z_history, z_next.unsqueeze(1)], dim=1)
            
            # Keep only last 10 frames for efficiency
            if z_history.shape[1] > 10:
                z_history = z_history[:, -10:]
        
        return torch.stack(generated, dim=1)


def edge_loss(pred, target):
    """Sobel edge loss for sharper predictions."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    # Convert to grayscale
    pred_gray = pred.mean(dim=1, keepdim=True)
    target_gray = target.mean(dim=1, keepdim=True)
    
    # Compute edges
    pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
    target_edge_x = F.conv2d(target_gray, sobel_x, padding=1)
    target_edge_y = F.conv2d(target_gray, sobel_y, padding=1)
    
    loss = F.l1_loss(pred_edge_x, target_edge_x) + F.l1_loss(pred_edge_y, target_edge_y)
    return loss


def train():
    device = get_device()
    print(f"Device: {device}")
    
    # Data - use longer sequences with larger batch for speed
    dataset = MovingMNISTDataset("data/moving_mnist.npy", seq_length=15)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Model
    model = EnhancedWorldModel(latent_dim=512).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    # Load checkpoint if exists
    start_epoch = 0
    ckpt_path = 'checkpoints/enhanced_world_model.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    print("\n" + "="*60)
    print("PHASE 1: Training Enhanced World Model (10 epochs)")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, 10):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_pred = 0
        epoch_edge = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            
            optimizer.zero_grad()
            out = model(frames)
            
            # Reconstruction loss
            loss_recon = F.binary_cross_entropy(out['recon'], frames)
            
            # Prediction loss
            frames_next = frames[:, 1:]
            loss_pred = F.binary_cross_entropy(out['pred'], frames_next)
            
            # Edge sharpness loss
            B, T, C, H, W = out['recon'].shape
            recon_flat = out['recon'].view(B * T, C, H, W)
            frames_flat = frames.view(B * T, C, H, W)
            loss_edge = edge_loss(recon_flat, frames_flat)
            
            # Latent consistency
            z_next_gt = out['z'][:, 1:]
            loss_latent = F.mse_loss(out['z_pred'], z_next_gt)
            
            # Total loss
            loss = loss_recon + loss_pred + 0.1 * loss_edge + 0.05 * loss_latent
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_pred += loss_pred.item()
            epoch_edge += loss_edge.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | recon: {loss_recon.item():.4f} | "
                      f"pred: {loss_pred.item():.4f} | edge: {loss_edge.item():.4f}")
        
        scheduler.step()
        n = len(loader)
        elapsed = time.time() - start
        avg_loss = epoch_loss / n
        
        print(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.4f} | "
              f"Recon: {epoch_recon/n:.4f} | Pred: {epoch_pred/n:.4f} | "
              f"Edge: {epoch_edge/n:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  ✓ Saved best model (loss: {best_loss:.4f})")
    
    print("\n" + "="*60)
    print("PHASE 2: Autoregressive Rollout Evaluation")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Get a sample
        sample = next(iter(loader))[:1].to(device)  # 1 sample, 20 frames
        
        # Use first 5 frames to predict next 25
        initial = sample[:, :5]
        generated = model.autoregressive_rollout(initial, num_steps=25)
        
        # Visualize
        inp = sample[0].cpu().numpy()  # 20 frames ground truth
        gen = generated[0].cpu().numpy()  # 25 generated frames
        init = initial[0].cpu().numpy()  # 5 initial frames
    
    # Plot rollout
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # Row 1-2: Ground truth (15 frames, show first 8 and next 7)
    for t in range(8):
        axes[0, t].imshow(inp[t].transpose(1, 2, 0))
        axes[0, t].axis('off')
        axes[0, t].set_title(f't={t}')
    for t in range(7):
        axes[1, t].imshow(inp[t+8].transpose(1, 2, 0) if t+8 < len(inp) else np.zeros((64,64,3)))
        axes[1, t].axis('off')
        axes[1, t].set_title(f't={t+8}')
    axes[1, 7].axis('off')
    
    # Row 3-4: Generated (25 frames, show first 16)
    for t in range(8):
        axes[2, t].imshow(gen[t].transpose(1, 2, 0))
        axes[2, t].axis('off')
        if t < 5:
            axes[2, t].set_title(f'seed→pred')
    for t in range(8):
        axes[3, t].imshow(gen[t+8].transpose(1, 2, 0))
        axes[3, t].axis('off')
    
    axes[0, 0].set_ylabel('GT', fontsize=10)
    axes[1, 0].set_ylabel('GT cont', fontsize=10)
    axes[2, 0].set_ylabel('Gen 0-7', fontsize=10)
    axes[3, 0].set_ylabel('Gen 8-15', fontsize=10)
    
    plt.suptitle('Autoregressive Rollout: 5 seed frames → 25 predicted frames', fontsize=14)
    plt.tight_layout()
    plt.savefig('autoregressive_rollout.png', dpi=150)
    print("Saved autoregressive_rollout.png")
    
    print("\n" + "="*60)
    print("PHASE 3: Physics Evaluation")
    print("="*60)
    
    evaluate_physics(model, loader, device)
    
    print("\n✅ Training Complete!")
    print(f"   - Best loss: {best_loss:.4f}")
    print(f"   - Model saved to: {ckpt_path}")
    print(f"   - Visualizations: autoregressive_rollout.png, physics_eval.png")


def evaluate_physics(model, loader, device):
    """Evaluate physics understanding: motion continuity, collision detection."""
    model.eval()
    
    metrics = {
        'motion_continuity': [],
        'position_accuracy': [],
        'collision_handling': [],
    }
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(loader):
            if batch_idx >= 10:  # Evaluate on 10 batches
                break
            
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            
            # Get predictions
            out = model(frames)
            pred = out['pred']  # (B, T-1, C, H, W)
            gt = frames[:, 1:]  # (B, T-1, C, H, W)
            
            # 1. Motion Continuity: Check if predictions change smoothly
            pred_diff = (pred[:, 1:] - pred[:, :-1]).abs().mean()
            gt_diff = (gt[:, 1:] - gt[:, :-1]).abs().mean()
            continuity = 1 - abs(pred_diff - gt_diff) / (gt_diff + 1e-6)
            metrics['motion_continuity'].append(continuity.item())
            
            # 2. Position Accuracy: Centroid tracking
            def get_centroid(img):
                # img: (C, H, W)
                gray = img.mean(dim=0)  # (H, W)
                total = gray.sum() + 1e-6
                h_idx = torch.arange(H, device=img.device).float()
                w_idx = torch.arange(W, device=img.device).float()
                cy = (gray.sum(dim=1) * h_idx).sum() / total
                cx = (gray.sum(dim=0) * w_idx).sum() / total
                return cx, cy
            
            pos_errors = []
            for b in range(min(B, 4)):
                for t in range(min(T-1, 10)):
                    px, py = get_centroid(pred[b, t])
                    gx, gy = get_centroid(gt[b, t])
                    err = ((px - gx)**2 + (py - gy)**2).sqrt()
                    pos_errors.append(err.item())
            metrics['position_accuracy'].append(np.mean(pos_errors))
            
            # 3. Collision Handling: Check prediction quality around motion changes
            # (Higher pixel variance indicates collision/bounce)
            variance = gt.var(dim=1).mean(dim=(1, 2, 3))  # Per-sample variance
            high_motion_mask = variance > variance.median()
            
            if high_motion_mask.sum() > 0:
                collision_mse = F.mse_loss(pred[high_motion_mask], gt[high_motion_mask])
                metrics['collision_handling'].append(collision_mse.item())
    
    # Report metrics
    print("\nPhysics Evaluation Results:")
    print(f"  Motion Continuity:  {np.mean(metrics['motion_continuity']):.4f} (1.0 = perfect)")
    print(f"  Position Error:     {np.mean(metrics['position_accuracy']):.2f} pixels")
    print(f"  Collision MSE:      {np.mean(metrics['collision_handling']):.4f} (lower = better)")
    
    # Visualize physics predictions
    model.eval()
    with torch.no_grad():
        sample = next(iter(loader))[:4].to(device)
        out = model(sample)
        
        fig, axes = plt.subplots(4, 10, figsize=(20, 8))
        
        for i in range(4):
            for t in range(5):
                # Ground truth
                axes[i, t].imshow(sample[i, t].cpu().numpy().transpose(1, 2, 0))
                axes[i, t].axis('off')
                if i == 0:
                    axes[i, t].set_title(f'GT t={t}')
                
                # Prediction
                axes[i, t+5].imshow(out['pred'][i, t].cpu().numpy().transpose(1, 2, 0))
                axes[i, t+5].axis('off')
                if i == 0:
                    axes[i, t+5].set_title(f'Pred t={t+1}')
        
        plt.suptitle('Physics Evaluation: Ground Truth vs Predictions', fontsize=14)
        plt.tight_layout()
        plt.savefig('physics_eval.png', dpi=150)
        print("Saved physics_eval.png")


if __name__ == "__main__":
    train()
