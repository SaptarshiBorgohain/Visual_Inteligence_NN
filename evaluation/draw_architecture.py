"""
Generate architecture diagram for the Sharp World Model.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',      # Light blue
        'encoder': '#BBDEFB',    # Blue
        'vq': '#CE93D8',         # Purple
        'decoder': '#C8E6C9',    # Green
        'gru': '#FFE0B2',        # Orange
        'discriminator': '#FFCDD2',  # Red
        'output': '#E8F5E9',     # Light green
        'skip': '#FFF9C4',       # Yellow
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2, color='black', style='->', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    
    # Title
    ax.text(9, 11.5, 'Sharp World Model Architecture', fontsize=16, fontweight='bold', ha='center')
    ax.text(9, 11, 'VQ-VAE + GAN + U-Net + GRU Dynamics', fontsize=12, ha='center', style='italic')
    
    # ========== ENCODER PATH (Left) ==========
    ax.text(2.5, 10.2, 'U-Net Encoder', fontsize=11, fontweight='bold', ha='center')
    
    # Input
    draw_box(1.5, 9, 2, 0.7, 'Input\n64×64×3', colors['input'], 8)
    
    # Encoder blocks
    draw_box(1.5, 7.8, 2, 0.7, 'Conv 64 + ResBlock\n32×32×64', colors['encoder'], 8)
    draw_box(1.5, 6.6, 2, 0.7, 'Conv 128 + ResBlock\n16×16×128', colors['encoder'], 8)
    draw_box(1.5, 5.4, 2, 0.7, 'Conv 256 + ResBlock\n8×8×256', colors['encoder'], 8)
    draw_box(1.5, 4.2, 2, 0.7, 'Conv 64\n4×4×64', colors['encoder'], 8)
    
    # Arrows down
    draw_arrow(2.5, 9, 2.5, 8.5)
    draw_arrow(2.5, 7.8, 2.5, 7.3)
    draw_arrow(2.5, 6.6, 2.5, 6.1)
    draw_arrow(2.5, 5.4, 2.5, 4.9)
    
    # ========== VECTOR QUANTIZATION (Center) ==========
    ax.text(7, 10.2, 'Vector Quantization', fontsize=11, fontweight='bold', ha='center')
    
    draw_box(5.5, 4.2, 3, 1.2, 'VQ Layer\n512 embeddings\ndim=64', colors['vq'], 9)
    
    # Arrow from encoder to VQ
    draw_arrow(3.5, 4.55, 5.5, 4.8)
    
    # Codebook visualization
    draw_box(5.5, 2.5, 3, 1.2, 'Codebook\n[512 × 64]\nDiscrete Latents', colors['vq'], 8)
    draw_arrow(7, 4.2, 7, 3.7, style='<->')
    
    # ========== GRU DYNAMICS (Center-Right) ==========
    ax.text(11.5, 10.2, 'GRU Dynamics', fontsize=11, fontweight='bold', ha='center')
    
    draw_box(10, 7, 3, 0.8, 'Flatten\n4×4×64 → 1024', colors['gru'], 8)
    draw_box(10, 5.8, 3, 0.8, 'GRU (2 layers)\nhidden=1024', colors['gru'], 8)
    draw_box(10, 4.6, 3, 0.8, 'MLP Predictor\n1024 → 1024', colors['gru'], 8)
    draw_box(10, 3.4, 3, 0.8, 'Reshape\n1024 → 4×4×64', colors['gru'], 8)
    
    # Arrows
    draw_arrow(8.5, 4.8, 10, 7.4)
    draw_arrow(11.5, 7, 11.5, 6.6)
    draw_arrow(11.5, 5.8, 11.5, 5.4)
    draw_arrow(11.5, 4.6, 11.5, 4.2)
    
    # ========== DECODER PATH (Right) ==========
    ax.text(15.5, 10.2, 'U-Net Decoder', fontsize=11, fontweight='bold', ha='center')
    
    draw_box(14.5, 4.2, 2, 0.7, 'DeConv 256\n8×8×256', colors['decoder'], 8)
    draw_box(14.5, 5.4, 2, 0.7, 'DeConv 128 + Res\n16×16×128', colors['decoder'], 8)
    draw_box(14.5, 6.6, 2, 0.7, 'DeConv 64 + Res\n32×32×64', colors['decoder'], 8)
    draw_box(14.5, 7.8, 2, 0.7, 'DeConv + Conv\n64×64×3', colors['decoder'], 8)
    
    # Output
    draw_box(14.5, 9, 2, 0.7, 'Output\n64×64×3', colors['output'], 8)
    
    # Arrows up
    draw_arrow(15.5, 4.9, 15.5, 5.4)
    draw_arrow(15.5, 6.1, 15.5, 6.6)
    draw_arrow(15.5, 7.3, 15.5, 7.8)
    draw_arrow(15.5, 8.5, 15.5, 9)
    
    # VQ to Decoder
    draw_arrow(8.5, 4.8, 14.5, 4.55)
    
    # GRU to Decoder (prediction path)
    draw_arrow(13, 3.8, 14.5, 4.55, color='orange', lw=2)
    
    # ========== SKIP CONNECTIONS ==========
    # Skip 1: 32x32
    ax.annotate('', xy=(14.5, 7), xytext=(3.5, 8.15),
               arrowprops=dict(arrowstyle='->', color='#FFC107', lw=2, 
                              connectionstyle='arc3,rad=0.2'))
    # Skip 2: 16x16
    ax.annotate('', xy=(14.5, 5.8), xytext=(3.5, 6.95),
               arrowprops=dict(arrowstyle='->', color='#FFC107', lw=2,
                              connectionstyle='arc3,rad=0.15'))
    # Skip 3: 8x8
    ax.annotate('', xy=(14.5, 4.6), xytext=(3.5, 5.75),
               arrowprops=dict(arrowstyle='->', color='#FFC107', lw=2,
                              connectionstyle='arc3,rad=0.1'))
    
    ax.text(9, 8.5, 'Skip Connections', fontsize=9, color='#FFC107', fontweight='bold', 
            ha='center', style='italic')
    
    # ========== DISCRIMINATOR ==========
    ax.text(15.5, 2.8, 'PatchGAN Discriminator', fontsize=10, fontweight='bold', ha='center')
    
    draw_box(14, 1.3, 3, 1.2, 'Conv Layers\n64→128→256→512\nOutput: Real/Fake', colors['discriminator'], 8)
    
    # Arrow from output to discriminator
    draw_arrow(15.5, 9, 16.5, 9, color='gray')
    ax.annotate('', xy=(15.5, 2.5), xytext=(16.5, 9),
               arrowprops=dict(arrowstyle='->', color='#E57373', lw=2,
                              connectionstyle='arc3,rad=-0.3'))
    
    # ========== LOSS FUNCTIONS ==========
    ax.text(2, 2.5, 'Loss Functions:', fontsize=10, fontweight='bold')
    ax.text(2, 2.0, '• Reconstruction: MSE(recon, input)', fontsize=9)
    ax.text(2, 1.6, '• Prediction: MSE(pred, next_frame)', fontsize=9)
    ax.text(2, 1.2, '• VQ Loss: commitment + codebook', fontsize=9)
    ax.text(2, 0.8, '• Adversarial: BCE(D(fake), 1)', fontsize=9)
    
    # ========== LEGEND ==========
    legend_elements = [
        mpatches.Patch(facecolor=colors['encoder'], edgecolor='black', label='Encoder'),
        mpatches.Patch(facecolor=colors['vq'], edgecolor='black', label='Vector Quantization'),
        mpatches.Patch(facecolor=colors['decoder'], edgecolor='black', label='Decoder'),
        mpatches.Patch(facecolor=colors['gru'], edgecolor='black', label='GRU Dynamics'),
        mpatches.Patch(facecolor=colors['discriminator'], edgecolor='black', label='Discriminator'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # ========== PARAMETER COUNT ==========
    ax.text(9, 0.5, 'Total Parameters: 22M (Generator: 19M, Discriminator: 3M)', 
            fontsize=10, ha='center', style='italic', 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('results/architecture_diagram.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved architecture diagram to results/architecture_diagram.png")
    plt.show()


if __name__ == "__main__":
    draw_architecture()
