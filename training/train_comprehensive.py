"""
Train World Model on ALL Physics Datasets
Comprehensive training for RL agent foundation
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
        self.data = np.load(npy_path)
        self.seq_length = min(seq_length, self.data.shape[1])
        self.name = os.path.basename(npy_path).replace('.npy', '')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx, :self.seq_length]
        seq = np.stack([seq, seq, seq], axis=1)
        seq = seq.astype(np.float32) / 255.0
        return torch.from_numpy(seq)


def train_comprehensive():
    device = get_device()
    print(f"Device: {device}")
    
    # Load ALL physics datasets
    print("\n" + "="*60)
    print("Loading ALL Physics Datasets")
    print("="*60)
    
    dataset_files = [
        'data/bouncing_balls_gravity.npy',
        'data/pendulum.npy',
        'data/spring_mass.npy',
        'data/multi_ball_collision.npy',
        'data/projectile_motion.npy',
        'data/double_pendulum.npy',
        'data/falling_objects.npy',
        'data/wave_motion.npy',
        'data/orbital_motion.npy',
        'data/moving_mnist.npy',  # Include MNIST too
    ]
    
    datasets = []
    for filepath in dataset_files:
        if os.path.exists(filepath):
            ds = PhysicsDataset(filepath, seq_length=5)
            datasets.append(ds)
            print(f"  ✓ {ds.name}: {len(ds)} sequences")
        else:
            print(f"  ✗ {filepath} not found")
    
    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=16, shuffle=True, num_workers=0)
    print(f"\nTotal: {len(combined)} sequences, {len(loader)} batches/epoch")
    
    # Load model
    model = SharpWorldModelV2(latent_channels=192, num_embeddings=512).to(device)
    
    # Try to load existing checkpoint
    ckpt_path = 'checkpoints/physics_world_model.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"\n✓ Loaded from {ckpt_path}")
    elif os.path.exists('checkpoints/sharp_world_model_v2.pt'):
        ckpt = torch.load('checkpoints/sharp_world_model_v2.pt', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"\n✓ Loaded from sharp_world_model_v2.pt")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    save_path = 'checkpoints/world_model_comprehensive.pt'
    os.makedirs('checkpoints', exist_ok=True)
    
    best_loss = float('inf')
    
    print("\n" + "="*60)
    print("COMPREHENSIVE PHYSICS TRAINING (10 epochs)")
    print("Learning: gravity, collisions, pendulums, waves, orbits...")
    print("="*60)
    
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_pred = 0
        start = time.time()
        
        for batch_idx, frames in enumerate(loader):
            frames = frames.to(device)
            
            optimizer.zero_grad()
            
            out = model(frames, use_dynamics=True)
            
            # Losses
            loss_recon = F.l1_loss(out['recon'], frames)
            loss_pred = F.l1_loss(out['pred'], frames[:, 1:])
            loss_vq = out['vq_loss']
            
            # Emphasize prediction for RL
            loss = 0.3 * loss_recon + 2.0 * loss_pred + 0.3 * loss_vq
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if device.type == 'mps':
                torch.mps.synchronize()
            
            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_pred += loss_pred.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | "
                      f"Recon: {loss_recon.item():.4f} | Pred: {loss_pred.item():.4f}")
        
        scheduler.step()
        
        n = len(loader)
        elapsed = time.time() - start
        avg_loss = epoch_loss / n
        
        print(f"\nEpoch {epoch+1}/10 | Loss: {avg_loss:.4f} | "
              f"Recon: {epoch_recon/n:.4f} | Pred: {epoch_pred/n:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  ✓ Saved (best loss: {best_loss:.4f})")
    
    print("\n" + "="*60)
    print("✅ Comprehensive Training Complete!")
    print(f"   Model: {save_path}")
    print(f"   Best Loss: {best_loss:.4f}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    train_comprehensive()
