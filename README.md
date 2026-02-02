# Visual Physics World Model

A PyTorch visual world model that learns physical dynamics from synthetic video sequences and uses imagination-based reinforcement learning (Dreamer-style) to train an agent.

## Highlights

- **World model:** VQ-VAE + ConvGRU dynamics (MPS-safe quantizer)
- **Training:** moving MNIST + 9 physics datasets (40k sequences total)
- **RL agent:** Dreamer-style actor-critic trained in imagination
- **Apple Silicon ready:** validated on MPS, also works on CUDA/CPU

## Project layout

```
visual_physics_world_model/
├── training/                    # Main training logic
│   ├── train_sharp_v2.py        # SharpWorldModelV2 (VQ-VAE + ConvGRU)
│   ├── train_comprehensive.py   # Multi-dataset physics training
│   ├── world_model_rl.py        # Dreamer-style RL agent
│   └── ... (earlier versions)
├── data_utils/                  # Dataset generation and loaders
│   ├── datasets.py              # PyTorch Dataset wrappers
│   ├── generate_physics_data.py # Basic generator
│   └── generate_diverse_datasets.py # Advanced physics scenes
├── evaluation/                  # Analysis and visualization
│   ├── eval_sharp.py            # Rollout evaluation
│   └── draw_architecture.py     # Architecture diagram generator
├── utils.py                     # Global device helpers
├── requirements.txt             # Python dependencies
├── data/                        # Local storage (gitignored)
├── checkpoints/                 # Model weights (gitignored)
└── results/                     # Output plots (gitignored)
```

## Quickstart

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Dataset Generation
python data_utils/generate_physics_data.py --dataset all
python data_utils/generate_diverse_datasets.py

# Training (World Model)
python training/train_sharp_v2.py
python training/train_comprehensive.py

# Training (RL Agent)
python training/world_model_rl.py
```

## Results

### World Model Performance
| Metric | Moving MNIST | Comprehensive Physics |
|--------|--------------|-----------------------|
| Recon Loss (L1) | 0.0021 | 0.0025 |
| Prediction Loss | 0.0142 | 0.0167 |
| Total Loss | 0.0310 | 0.0348 |
| Training Time (4 epochs) | ~3 hours | ~3.5 hours |

### RL Agent (Imagination Training)
The agent learns to interact with the environment purely within the world model's latent imagination.
- **Training:** 50 iterations of "dreaming"
- **Imagination Reward:** reached ~-0.01 (normalized distance to target)
- **Real Environment Test:** successfully executed policy on real physics engine frames
- **Total Episode Reward:** -24.67 (negative = closer to goal)

### Generated Visualizations

#### World Model Results
- `sharp_model_v2_result.png` - Sharp reconstruction on Moving MNIST
- `physics_bouncing_balls_result.png` - Physics prediction (bouncing balls)
- `physics_pendulum_result.png` - Physics prediction (pendulum)
- `physics_spring_mass_result.png` - Physics prediction (spring-mass)
- `autoregressive_rollout.png` - Multi-step predictions

#### RL Agent Results
- `rl_training.png` - Actor/Critic/Reward training curves
- `rl_trajectory.png` - Agent trajectory in real environment

#### Architecture & Data
- `architecture_diagram.png` - Full model architecture visualization
- `bouncing_balls_preview.png` - Sample physics dataset
- `pendulum_preview.png` - Pendulum dynamics samples
- `spring_mass_preview.png` - Spring-mass system samples

## Datasets included

- Moving MNIST (10k)
- Bouncing balls (gravity) (5k)
- Pendulum (5k)
- Spring-mass (5k)
- Multi-ball collision (3k)
- Projectile motion (3k)
- Double pendulum (2k)
- Falling objects (3k)
- Wave motion (2k)
- Orbital motion (2k)

## Notes on large files

The `data/`, `checkpoints/`, and `results/` directories are intentionally **gitignored**. Generate data and models locally, or add your own artifact hosting if you want to publish them.

## Hardware

- Tested on Apple Silicon (MPS)
- Also supports CUDA and CPU

## License

MIT. See `LICENSE`.
