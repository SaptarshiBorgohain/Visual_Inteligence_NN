"""
Generate architecture diagram for the Sharp World Model.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E3F2FD',      # Light blue
        'encoder': '#BBDEFB',    # Blue
        'vq': '#CE93D8',         # Purple
        'decoder': '#C8E6C9',    # Green
        'gru': '#FFE0B2',        # Orange
        'rl': '#FFCC80',         # Darker Orange/Peach
        'output': '#E8F5E9',     # Light green
        'skip': '#FFF9C4',       # Yellow
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    
    def draw_arrow(x1, y1, x2, y2, color='black', style='->', lw=1.5, connectionstyle=None, linestyle='-'):
        arrow_dict = dict(arrowstyle=style, color=color, lw=lw, linestyle=linestyle)
        if connectionstyle:
            arrow_dict['connectionstyle'] = connectionstyle
            
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_dict)
    
    # Title
    ax.text(10, 13.5, 'Visual Physics World Model V2', fontsize=20, fontweight='bold', ha='center')
    ax.text(10, 13, 'VQ-VAE + ConvGRU Dynamics + Imagination RL', fontsize=14, ha='center', style='italic')
    
    # ========== ENCODER PATH (Left) ==========
    ax.text(3, 11.5, 'Convolutional Encoder', fontsize=12, fontweight='bold', ha='center')
    
    # Input
    draw_box(2, 10, 2, 0.8, 'Input Frame\n64×64×3', colors['input'], 9)
    
    # Encoder blocks
    draw_box(2, 8.5, 2, 0.8, 'Conv 64\n32×32×64', colors['encoder'], 9)
    draw_box(2, 7.0, 2, 0.8, 'Conv 128\n16×16×128', colors['encoder'], 9)
    draw_box(2, 5.5, 2, 0.8, 'Conv 192\n8×8×192', colors['encoder'], 9)
    
    # Arrows down
    draw_arrow(3, 10, 3, 9.3)
    draw_arrow(3, 8.5, 3, 7.8)
    draw_arrow(3, 7.0, 3, 6.3)
    
    # ========== VECTOR QUANTIZATION (Center-Left) ==========
    ax.text(6.5, 11.5, 'Vector Quantization (MPS-Safe)', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(5, 5.3, 3, 1.2, 'VQ Layer\n512 embeddings\ndim=192', colors['vq'], 10)
    
    # Arrow from encoder to VQ
    draw_arrow(4, 5.9, 5, 5.9)
    
    # Codebook visualization
    draw_box(5, 3.5, 3, 1.2, 'Codebook\n[512 × 192]\nDiscrete Latents', colors['vq'], 9)
    draw_arrow(6.5, 5.3, 6.5, 4.7, style='<->')
    
    # ========== DYNAMICS MODEL (Center) ==========
    ax.text(10.5, 11.5, 'Dynamics Model', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(9, 7.5, 3, 1.0, 'Layer 2: ConvGRU\n8×8×192', colors['gru'], 9)
    draw_box(9, 6.0, 3, 1.0, 'Layer 1: ConvGRU\n8×8×192', colors['gru'], 9)
    
    # Action Input
    draw_box(9, 4.5, 3, 0.6, 'Action Input\n(Broadcasted)', '#FFF3E0', 9)
    draw_arrow(10.5, 5.1, 10.5, 6.0)
    
    # VQ to Dynamics
    draw_arrow(8, 5.9, 9, 6.5, connectionstyle='arc3,rad=0.1')
    
    # Recurrent loop
    ax.annotate('State t-1', xy=(10.5, 7.5), xytext=(12.5, 8.5),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2, connectionstyle='arc3,rad=0.3'),
                fontsize=9, color='#E65100', ha='center')

    # Dynamics internal arrows
    draw_arrow(10.5, 7.0, 10.5, 7.5)
    
    # ========== DECODER PATH (Right) ==========
    ax.text(15, 11.5, 'Convolutional Decoder', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(14, 5.5, 2, 0.8, 'DeConv 192\n8×8×192', colors['decoder'], 9)
    draw_box(14, 7.0, 2, 0.8, 'DeConv 128\n16×16×128', colors['decoder'], 9)
    draw_box(14, 8.5, 2, 0.8, 'DeConv 64\n32×32×64', colors['decoder'], 9)
    
    # Output
    draw_box(14, 10, 2, 0.8, 'Output Frame\n64×64×3', colors['output'], 9)
    
    # Arrows up
    draw_arrow(15, 6.3, 15, 7.0)
    draw_arrow(15, 7.8, 15, 8.5)
    draw_arrow(15, 9.3, 15, 10.0)
    
    # Dynamics to Decoder (Reconstruction/Prediction)
    draw_arrow(12, 8, 14, 6, connectionstyle='arc3,rad=-0.1', color='orange', lw=2)
    # VQ to Decoder (Direct Reconstruction)
    draw_arrow(8, 5.9, 14, 5.9, connectionstyle='arc3,rad=-0.05', linestyle='dashed')
    
    # ========== SKIP CONNECTIONS ==========
    # Skip 1: 16x16
    ax.annotate('Skip', xy=(14, 7.4), xytext=(4, 7.4),
               arrowprops=dict(arrowstyle='->', color='#FBC02D', lw=1.5, 
                              connectionstyle='arc3,rad=0.15'), fontsize=8, color='#FBC02D', ha='center')
    # Skip 2: 32x32
    ax.annotate('Skip', xy=(14, 8.9), xytext=(4, 8.9),
               arrowprops=dict(arrowstyle='->', color='#FBC02D', lw=1.5,
                              connectionstyle='arc3,rad=0.2'), fontsize=8, color='#FBC02D', ha='center')
    
    # ========== RL AGENT HEADS (Bottom) ==========
    ax.text(10, 3.0, 'Dreamer RL Agent (Latent Imagination)', fontsize=12, fontweight='bold', ha='center')
    
    draw_box(6, 1.5, 2, 1.0, 'Actor (π)\nAction Dist.', colors['rl'], 9)
    draw_box(9, 1.5, 2, 1.0, 'Critic (V)\nValue Est.', colors['rl'], 9)
    draw_box(12, 1.5, 2, 1.0, 'Reward (R)\nPredictor', colors['rl'], 9)
    
    # Arrows from Dynamics to RL
    draw_arrow(10.5, 6.0, 7, 2.5, connectionstyle='arc3,rad=-0.1', linestyle='dashed', color='gray') # To Actor
    draw_arrow(10.5, 6.0, 10, 2.5, linestyle='dashed', color='gray') # To Critic
    draw_arrow(10.5, 6.0, 13, 2.5, connectionstyle='arc3,rad=0.1', linestyle='dashed', color='gray') # To Reward
    
    
    # ========== LOSS FUNCTIONS ==========
    ax.text(1, 4.0, 'Training Objectives:', fontsize=11, fontweight='bold')
    ax.text(1, 3.5, '• Reconstruction (L1)', fontsize=10)
    ax.text(1, 3.0, '• Latent Prediction (MSE)', fontsize=10)
    ax.text(1, 2.5, '• VQ Commitment (No EMA)', fontsize=10)
    ax.text(1, 2.0, '• Actor Entropy + Return', fontsize=10)
    ax.text(1, 1.5, '• Critic/Reward MSE', fontsize=10)
    
    # ========== LEGEND ==========
    legend_elements = [
        mpatches.Patch(facecolor=colors['encoder'], edgecolor='black', label='Encoder'),
        mpatches.Patch(facecolor=colors['vq'], edgecolor='black', label='Vector Quantizer'),
        mpatches.Patch(facecolor=colors['gru'], edgecolor='black', label='ConvGRU Dynamics'),
        mpatches.Patch(facecolor=colors['decoder'], edgecolor='black', label='Decoder'),
        mpatches.Patch(facecolor=colors['rl'], edgecolor='black', label='RL Heads'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # ========== PARAMETER COUNT ==========
    ax.text(10, 0.5, 'Total Parameters: 9.5M (Encoder: 1.2M, Decoder: 1.1M, ConvGRU: 6.0M, VQ+RL: ~1.2M)', 
            fontsize=11, ha='center', style='italic', 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('results/architecture_diagram.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Saved architecture diagram to results/architecture_diagram.png")
    plt.show()


if __name__ == "__main__":
    draw_architecture()
