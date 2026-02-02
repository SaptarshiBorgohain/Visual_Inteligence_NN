"""
World Model RL Agent (DreamerV3 Style)
Uses the trained world model as an imagination engine for training an RL agent.

Key Components:
1. World Model (trained VQ-VAE + ConvGRU) - predicts future states
2. Actor Network - outputs actions
3. Critic Network - estimates values
4. Imagination Training - train actor/critic in "dreams"
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.train_sharp_v2 import SharpWorldModelV2
from utils import get_device


# ============================================================
# ACTOR NETWORK - Outputs actions given latent state
# ============================================================
class Actor(nn.Module):
    """Policy network that outputs actions from latent states."""
    
    def __init__(self, latent_dim=192, hidden_dim=256, action_dim=4):
        super().__init__()
        
        # Process 8x8x192 latent
        self.conv = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, 2, 1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Action head (continuous actions)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, latent):
        """
        Args:
            latent: (B, C, H, W) latent state from world model
        Returns:
            action_dist: Normal distribution over actions
        """
        x = self.conv(latent)
        x = self.fc(x)
        
        mean = self.action_mean(x)
        std = self.action_logstd.exp().expand_as(mean)
        
        return torch.distributions.Normal(mean, std)
    
    def get_action(self, latent, deterministic=False):
        dist = self.forward(latent)
        if deterministic:
            return dist.mean
        return dist.rsample()


# ============================================================
# CRITIC NETWORK - Estimates value of latent state
# ============================================================
class Critic(nn.Module):
    """Value network that estimates returns from latent states."""
    
    def __init__(self, latent_dim=192, hidden_dim=256):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, latent):
        x = self.conv(latent)
        return self.fc(x)


# ============================================================
# REWARD PREDICTOR - Learns to predict rewards
# ============================================================
class RewardPredictor(nn.Module):
    """Predicts reward from latent state and action."""
    
    def __init__(self, latent_dim=192, action_dim=4, hidden_dim=128):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, latent, action):
        x = self.conv(latent)
        x = torch.cat([x, action], dim=-1)
        return self.fc(x)


# ============================================================
# WORLD MODEL RL AGENT
# ============================================================
class WorldModelAgent:
    """
    DreamerV3-style agent that learns in imagination.
    
    Training loop:
    1. Collect real experience -> train world model
    2. Imagine trajectories using world model
    3. Train actor-critic on imagined trajectories
    4. Execute actions in real environment
    """
    
    def __init__(self, world_model_path=None, device=None):
        self.device = device or get_device()
        
        # World Model (frozen during RL training)
        self.world_model = SharpWorldModelV2(latent_channels=192, num_embeddings=512).to(self.device)
        if world_model_path and os.path.exists(world_model_path):
            ckpt = torch.load(world_model_path, map_location=self.device, weights_only=False)
            self.world_model.load_state_dict(ckpt['model'])
            print(f"✓ Loaded world model from {world_model_path}")
        self.world_model.eval()
        
        # Actor-Critic
        self.actor = Actor(latent_dim=192, hidden_dim=256, action_dim=4).to(self.device)
        self.critic = Critic(latent_dim=192, hidden_dim=256).to(self.device)
        self.reward_predictor = RewardPredictor(latent_dim=192, action_dim=4).to(self.device)
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.reward_opt = optim.Adam(self.reward_predictor.parameters(), lr=3e-4)
        
        # Training params
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.imagination_horizon = 15
        
        # Stats
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'imagined_reward': [],
        }
    
    def encode_observation(self, obs):
        """Encode observation to latent state using world model."""
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            z, skips, _, _ = self.world_model.encode(obs)
        return z, skips
    
    def imagine_trajectory(self, initial_latent, horizon=15):
        """
        Imagine a trajectory using world model dynamics.
        This is where the "dreaming" happens!
        """
        latents = [initial_latent]
        actions = []
        log_probs = []
        
        z = initial_latent
        hidden = None
        
        for t in range(horizon):
            # Get action from actor
            action_dist = self.actor(z)
            action = action_dist.rsample()
            log_prob = action_dist.log_prob(action).sum(-1)
            
            actions.append(action)
            log_probs.append(log_prob)
            
            # Predict next latent using world model dynamics
            with torch.no_grad():
                z_next, hidden = self.world_model.dynamics.predict_next(z, hidden)
            
            latents.append(z_next)
            z = z_next
        
        return {
            'latents': torch.stack(latents, dim=1),  # (B, T+1, C, H, W)
            'actions': torch.stack(actions, dim=1),   # (B, T, action_dim)
            'log_probs': torch.stack(log_probs, dim=1),  # (B, T)
        }
    
    def compute_imagined_rewards(self, latents, actions, goal_latent=None):
        """
        Compute rewards for imagined trajectory.
        Can use learned reward predictor or hand-crafted rewards.
        """
        B, T, C, H, W = latents[:, :-1].shape
        
        # Flatten for reward prediction
        latents_flat = latents[:, :-1].reshape(B * T, C, H, W)
        actions_flat = actions.reshape(B * T, -1)
        
        # Predicted rewards
        rewards = self.reward_predictor(latents_flat, actions_flat)
        rewards = rewards.reshape(B, T)
        
        # Optional: Add goal-reaching bonus
        if goal_latent is not None:
            final_latent = latents[:, -1]
            goal_dist = F.mse_loss(final_latent, goal_latent, reduction='none').mean(dim=(1,2,3))
            goal_bonus = -goal_dist.unsqueeze(1).expand(-1, T) * 0.1
            rewards = rewards + goal_bonus
        
        return rewards
    
    def compute_returns(self, rewards, values):
        """Compute lambda-returns for actor-critic training."""
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        
        # GAE-style returns
        last_value = values[:, -1]
        last_return = last_value
        
        for t in reversed(range(T)):
            returns[:, t] = rewards[:, t] + self.gamma * last_return
            last_return = returns[:, t]
        
        return returns
    
    def train_imagination(self, initial_obs, n_iterations=100):
        """
        Main training loop - train actor-critic in imagination.
        
        Args:
            initial_obs: Starting observation (B, C, H, W)
            n_iterations: Number of training iterations
        """
        print("\n" + "="*60)
        print("Training Agent in Imagination")
        print("="*60)
        
        for iteration in range(n_iterations):
            # Encode initial observation
            z0, _ = self.encode_observation(initial_obs)
            
            # Imagine trajectory
            trajectory = self.imagine_trajectory(z0, horizon=self.imagination_horizon)
            
            # Compute values for all latents
            latents = trajectory['latents']
            B, T_plus_1, C, H, W = latents.shape
            
            values = self.critic(latents.reshape(-1, C, H, W))
            values = values.reshape(B, T_plus_1)
            
            # Compute rewards
            rewards = self.compute_imagined_rewards(
                trajectory['latents'], 
                trajectory['actions']
            )
            
            # Compute returns
            returns = self.compute_returns(rewards, values)
            
            # ================
            # Train Critic
            # ================
            critic_loss = F.mse_loss(values[:, :-1], returns.detach())
            
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            
            # ================
            # Train Actor (policy gradient)
            # ================
            # Re-compute trajectory for actor gradients
            trajectory = self.imagine_trajectory(z0.detach(), horizon=self.imagination_horizon)
            
            with torch.no_grad():
                values_detached = self.critic(trajectory['latents'].reshape(-1, C, H, W))
                values_detached = values_detached.reshape(B, T_plus_1)
                rewards = self.compute_imagined_rewards(
                    trajectory['latents'], 
                    trajectory['actions']
                )
                returns = self.compute_returns(rewards, values_detached)
                advantages = returns - values_detached[:, :-1]
            
            # Policy loss
            actor_loss = -(trajectory['log_probs'] * advantages).mean()
            
            # Entropy bonus for exploration
            entropy = -trajectory['log_probs'].mean()
            actor_loss = actor_loss - 0.01 * entropy
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # Log stats
            self.training_stats['actor_loss'].append(actor_loss.item())
            self.training_stats['critic_loss'].append(critic_loss.item())
            self.training_stats['imagined_reward'].append(rewards.mean().item())
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iter {iteration+1}/{n_iterations} | "
                      f"Actor: {actor_loss.item():.4f} | "
                      f"Critic: {critic_loss.item():.4f} | "
                      f"Reward: {rewards.mean().item():.4f}")
        
        print("\n✓ Imagination training complete!")
        return self.training_stats
    
    def act(self, observation, deterministic=False):
        """Get action for real environment."""
        with torch.no_grad():
            z, _ = self.encode_observation(observation)
            action = self.actor.get_action(z, deterministic=deterministic)
            return action.detach().cpu().numpy()
    
    def save(self, path):
        """Save agent checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'training_stats': self.training_stats,
        }, path)
        print(f"✓ Agent saved to {path}")
    
    def load(self, path):
        """Load agent checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.reward_predictor.load_state_dict(ckpt['reward_predictor'])
        self.training_stats = ckpt['training_stats']
        print(f"✓ Agent loaded from {path}")


# ============================================================
# SIMPLE PHYSICS ENVIRONMENT (for testing)
# ============================================================
class SimplePhysicsEnv:
    """
    Simple 2D physics environment for testing the agent.
    Task: Control a ball to reach a target position.
    """
    
    def __init__(self, size=64):
        self.size = size
        self.reset()
    
    def reset(self):
        self.ball_pos = np.array([self.size//4, self.size//2], dtype=np.float32)
        self.ball_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.target_pos = np.array([3*self.size//4, self.size//2], dtype=np.float32)
        return self._get_obs()
    
    def step(self, action):
        # Action: [force_x, force_y, ...]
        force = np.array(action[:2]) * 2.0
        
        # Physics
        self.ball_vel += force
        self.ball_vel *= 0.95  # Friction
        self.ball_pos += self.ball_vel
        
        # Boundaries
        self.ball_pos = np.clip(self.ball_pos, 5, self.size - 5)
        
        # Reward: negative distance to target
        dist = np.linalg.norm(self.ball_pos - self.target_pos)
        reward = -dist / self.size
        
        # Done if reached target
        done = dist < 5
        if done:
            reward += 10.0
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        """Render observation as image."""
        frame = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Draw target
        y, x = np.ogrid[:self.size, :self.size]
        target_mask = (x - self.target_pos[0])**2 + (y - self.target_pos[1])**2 <= 25
        frame[target_mask] = 128
        
        # Draw ball
        ball_mask = (x - self.ball_pos[0])**2 + (y - self.ball_pos[1])**2 <= 16
        frame[ball_mask] = 255
        
        # Convert to RGB tensor
        frame_rgb = np.stack([frame, frame, frame], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(frame_rgb)


# ============================================================
# DEMO
# ============================================================
def demo_rl_agent():
    """Demonstrate the RL agent training."""
    device = get_device()
    print(f"Device: {device}")
    
    # Find world model
    model_paths = [
        'checkpoints/world_model_comprehensive.pt',
        'checkpoints/physics_world_model.pt',
        'checkpoints/sharp_world_model_v2.pt',
    ]
    
    world_model_path = None
    for path in model_paths:
        if os.path.exists(path):
            world_model_path = path
            break
    
    if world_model_path is None:
        print("No world model found! Please train a world model first.")
        return
    
    # Create agent
    agent = WorldModelAgent(world_model_path=world_model_path, device=device)
    
    # Create environment
    env = SimplePhysicsEnv(size=64)
    
    # Get initial observation
    obs = env.reset().to(device)
    
    # Train in imagination
    print("\n" + "="*60)
    print("WORLD MODEL RL TRAINING DEMO")
    print("="*60)
    print("\nThe agent will now 'dream' many trajectories")
    print("and learn a policy purely in imagination!")
    
    # Expand obs for batch
    obs_batch = obs.unsqueeze(0).expand(8, -1, -1, -1)  # Batch of 8
    
    # Train
    stats = agent.train_imagination(obs_batch, n_iterations=50)
    
    # Save agent
    os.makedirs('checkpoints', exist_ok=True)
    agent.save('checkpoints/rl_agent.pt')
    
    # Visualize training
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(stats['actor_loss'])
    axes[0].set_title('Actor Loss')
    axes[0].set_xlabel('Iteration')
    
    axes[1].plot(stats['critic_loss'])
    axes[1].set_title('Critic Loss')
    axes[1].set_xlabel('Iteration')
    
    axes[2].plot(stats['imagined_reward'])
    axes[2].set_title('Imagined Reward')
    axes[2].set_xlabel('Iteration')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/rl_training.png', dpi=150)
    plt.close()
    print("\n✓ Saved results/rl_training.png")
    
    # Test agent in real environment
    print("\n" + "="*60)
    print("Testing Agent in Real Environment")
    print("="*60)
    
    obs = env.reset()
    total_reward = 0
    trajectory = [obs.numpy().transpose(1, 2, 0)]
    
    for step in range(50):
        action = agent.act(obs.unsqueeze(0).to(device), deterministic=True)
        obs, reward, done, _ = env.step(action[0])
        total_reward += reward
        trajectory.append(obs.numpy().transpose(1, 2, 0))
        
        if done:
            print(f"  Reached target in {step+1} steps!")
            break
    
    print(f"  Total reward: {total_reward:.2f}")
    
    # Visualize trajectory
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        idx = i * len(trajectory) // 10
        if idx < len(trajectory):
            ax.imshow(trajectory[idx])
            ax.set_title(f't={idx}')
            ax.axis('off')
    
    plt.suptitle('Agent Trajectory in Real Environment', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/rl_trajectory.png', dpi=150)
    plt.close()
    print("✓ Saved results/rl_trajectory.png")
    
    print("\n" + "="*60)
    print("✅ RL Agent Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo_rl_agent()
