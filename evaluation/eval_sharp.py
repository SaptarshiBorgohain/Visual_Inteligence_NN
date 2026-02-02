"""Quick evaluation of sharp model."""
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_utils.datasets import MovingMNISTDataset
from utils import get_device
from training.train_sharp import SharpWorldModel

device = get_device()
print(f"Device: {device}")

# Load model
model = SharpWorldModel(latent_channels=64, num_embeddings=512).to(device)
ckpt = torch.load('checkpoints/sharp_world_model.pt', map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded from epoch {ckpt['epoch']+1}, loss: {ckpt['loss']:.4f}")

# Data
dataset = MovingMNISTDataset("data/moving_mnist.npy", seq_length=10)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

with torch.no_grad():
    sample = next(iter(loader)).to(device)
    out = model(sample)
    
    # Autoregressive
    initial = sample[:1, :5]
    generated = model.autoregressive_rollout(initial, num_steps=15)

# Visualize
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

plt.suptitle('Sharp World Model (VQ-VAE + GAN) - After 5 Epochs', fontsize=14)
plt.tight_layout()
plt.savefig('sharp_model_result.png', dpi=150)
print("Saved sharp_model_result.png")
