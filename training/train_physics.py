"""
Train Sharp World Model V2 on Physics Datasets
Fine-tune from Moving MNIST checkpoint to learn physics dynamics
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import time
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train_sharp_v2 import SharpWorldModelV2
from utils import get_device


class PhysicsDataset(Dataset):
    """Dataset for grayscale physics simulations."""
    
    def __init__(self, npy_path, seq_length=5):
        self.data = np.load(npy_path)  # (N, T, H, W)
        self.seq_length = min(seq_length, self.data.shape[1])
        print(f"  Loaded {npy_path}: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx, :self.seq_length]  # (T, H, W)
        # Convert to RGB and normalize
        seq = np.stack([seq, seq, seq], axis=1)  # (T, 3, H, W)
        seq = seq.astype(np.float32) / 255.0
        return torch.from_numpy(seq)


def train_physics():
    device = get_device()
    print(f"Device: {device}")
    
    # Load all physics datasets
    print("\nLoading physics datasets...")
    datasets = []
    
    # Bouncing balls with gravity
    if os.path.exists("data/bouncing_balls_gravity.npy"):
        ds = PhysicsDataset("data/bouncing_balls_gravity.npy", seq_length=5)
        datasets.append(ds)
    
    # Pendulum
    if os.path.exists("data/pendulum.npy"):
        ds = PhysicsDataset("data/pendulum.npy", seq_length=5)
        datasets.append(ds)
    
    # Spring-mass
    if os.path.exists("data/spring_mass.npy"):
        ds = PhysicsDataset("data/spring_mass.npy", seq_length=5)
        datasets.append(ds)
    
    if not datasets:
        print("No physics datasets found!")
        return
    
    # Combine all datasets
    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=8, shuffle=True, num_workers=0)
    print(f"\nTotal: {len(combined)} sequences, {len(loader)} batches")
    
    # Load pretrained model
    model = SharpWorldModelV2(latent_channels=192, num_embeddings=512).to(device)
    
    ckpt_path = 'checkpoints/sharp_world_model_v2.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"\n✓ Loaded pretrained model from {ckpt_path}")
    else:
        print("\n⚠ No pretrained model found, training from scratch")
    
    # Optimizer - lower LR for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    physics_ckpt_path = 'checkpoints/physics_world_model.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    best_loss = float('inf')
    
    print("\n" + "="*60)
    print("PHYSICS TRAINING (5 epochs)")
    print("Learning physical dynamics: gravity, oscillation, springs")
    print("="*60)
    
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_pred = 0
        epoch_vq = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            B, T, C, H, W = frames.shape
            
            optimizer.zero_grad()
            
            # Forward with dynamics
            out = model(frames, use_dynamics=True)
            
            # Reconstruction loss (L1)
            loss_recon = F.l1_loss(out['recon'], frames)
            
            # Prediction loss - predict next frame from current
            loss_pred = F.l1_loss(out['pred'], frames[:, 1:])
            
            # VQ loss
            loss_vq = out['vq_loss']
            
            # Total loss - emphasize prediction for physics learning
            loss = 0.5 * loss_recon + 1.5 * loss_pred + 0.5 * loss_vq
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # MPS sync
            if device.type == 'mps':
                torch.mps.synchronize()
            
            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_pred += loss_pred.item()
            epoch_vq += loss_vq.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | "
                      f"Recon: {loss_recon.item():.4f} | "
                      f"Pred: {loss_pred.item():.4f} | "
                      f"VQ: {loss_vq.item():.4f}")
        
        scheduler.step()
        
        n = len(loader)
        elapsed = time.time() - start
        avg_loss = epoch_loss / n
        
        print(f"\nEpoch {epoch+1}/5 | Loss: {avg_loss:.4f} | "
              f"Recon: {epoch_recon/n:.4f} | Pred: {epoch_pred/n:.4f} | "
              f"VQ: {epoch_vq/n:.4f} | Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, physics_ckpt_path)
            print(f"  ✓ Saved to {physics_ckpt_path}")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    print("\n" + "="*60)
    print("Generating Physics Visualizations")
    print("="*60)
    
    model.eval()
    os.makedirs('results', exist_ok=True)
    
    # Test on each physics type
    physics_names = ['bouncing_balls', 'pendulum', 'spring_mass']
    physics_files = ['data/bouncing_balls_gravity.npy', 'data/pendulum.npy', 'data/spring_mass.npy']
    
    for name, filepath in zip(physics_names, physics_files):
        if not os.path.exists(filepath):
            continue
            
        print(f"\nVisualizing {name}...")
        
        ds = PhysicsDataset(filepath, seq_length=5)
        sample = ds[0].unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(sample, use_dynamics=True)
            
            # Autoregressive rollout - start from first 2 frames
            initial = sample[:, :2]
            generated = model.autoregressive_rollout(initial, num_steps=10)
        
        # Plot
        fig, axes = plt.subplots(3, 5, figsize=(12, 7))
        
        # Row 1: Ground truth
        for t in range(5):
            img = sample[0, t].cpu().numpy().transpose(1, 2, 0)
            axes[0, t].imshow(img)
            axes[0, t].axis('off')
            axes[0, t].set_title(f't={t}')
        
        # Row 2: Reconstruction
        for t in range(5):
            img = out['recon'][0, t].cpu().numpy().transpose(1, 2, 0)
            axes[1, t].imshow(img)
            axes[1, t].axis('off')
        
        # Row 3: Autoregressive prediction
        for t in range(min(5, generated.shape[1])):
            img = generated[0, t].cpu().numpy().transpose(1, 2, 0)
            axes[2, t].imshow(img)
            axes[2, t].axis('off')
        
        axes[0, 0].set_ylabel('Ground Truth', fontsize=10)
        axes[1, 0].set_ylabel('Reconstruction', fontsize=10)
        axes[2, 0].set_ylabel('Predicted', fontsize=10)
        
        plt.suptitle(f'Physics World Model: {name.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'results/physics_{name}_result.png', dpi=150)
        plt.close()
        print(f"  Saved results/physics_{name}_result.png")
    
    print("\n" + "="*60)
    print("✅ Physics Training Complete!")
    print(f"   Model saved to: {physics_ckpt_path}")
    print(f"   Best loss: {best_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    train_physics()
