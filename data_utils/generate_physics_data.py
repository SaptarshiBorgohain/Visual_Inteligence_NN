"""
Generate physics datasets with more complex scenarios.
1. Bouncing Balls with Gravity
2. Multiple objects with collisions
3. Pendulum motion
"""
import numpy as np
import os
from PIL import Image, ImageDraw
import math


def generate_bouncing_balls_gravity(
    num_sequences=5000,
    seq_length=30,
    img_size=64,
    num_balls=2,
    ball_radius=5,
    gravity=0.3,
    friction=0.99,
    save_path="data/bouncing_balls_gravity.npy"
):
    """
    Generate bouncing balls with gravity physics.
    Balls fall down, bounce off walls and each other.
    """
    print(f"Generating {num_sequences} bouncing ball sequences with gravity...")
    
    data = np.zeros((num_sequences, seq_length, img_size, img_size), dtype=np.uint8)
    
    for seq in range(num_sequences):
        # Initialize balls
        balls = []
        for _ in range(num_balls):
            x = np.random.uniform(ball_radius + 5, img_size - ball_radius - 5)
            y = np.random.uniform(ball_radius + 5, img_size // 2)  # Start in upper half
            vx = np.random.uniform(-2, 2)
            vy = np.random.uniform(-1, 1)
            balls.append({'x': x, 'y': y, 'vx': vx, 'vy': vy})
        
        for t in range(seq_length):
            # Create frame
            img = Image.new('L', (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            
            # Update and draw each ball
            for ball in balls:
                # Apply gravity
                ball['vy'] += gravity
                
                # Apply friction
                ball['vx'] *= friction
                ball['vy'] *= friction
                
                # Update position
                ball['x'] += ball['vx']
                ball['y'] += ball['vy']
                
                # Bounce off walls
                if ball['x'] <= ball_radius:
                    ball['x'] = ball_radius
                    ball['vx'] = -ball['vx'] * 0.9
                if ball['x'] >= img_size - ball_radius:
                    ball['x'] = img_size - ball_radius
                    ball['vx'] = -ball['vx'] * 0.9
                if ball['y'] <= ball_radius:
                    ball['y'] = ball_radius
                    ball['vy'] = -ball['vy'] * 0.9
                if ball['y'] >= img_size - ball_radius:
                    ball['y'] = img_size - ball_radius
                    ball['vy'] = -ball['vy'] * 0.9
                
                # Draw ball
                draw.ellipse([
                    ball['x'] - ball_radius, ball['y'] - ball_radius,
                    ball['x'] + ball_radius, ball['y'] + ball_radius
                ], fill=255)
            
            # Ball-ball collision
            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    dx = balls[j]['x'] - balls[i]['x']
                    dy = balls[j]['y'] - balls[i]['y']
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    if dist < 2 * ball_radius and dist > 0:
                        # Elastic collision
                        nx, ny = dx / dist, dy / dist
                        
                        # Relative velocity
                        dvx = balls[i]['vx'] - balls[j]['vx']
                        dvy = balls[i]['vy'] - balls[j]['vy']
                        dvn = dvx * nx + dvy * ny
                        
                        # Update velocities
                        balls[i]['vx'] -= dvn * nx
                        balls[i]['vy'] -= dvn * ny
                        balls[j]['vx'] += dvn * nx
                        balls[j]['vy'] += dvn * ny
                        
                        # Separate balls
                        overlap = 2 * ball_radius - dist
                        balls[i]['x'] -= overlap * nx / 2
                        balls[i]['y'] -= overlap * ny / 2
                        balls[j]['x'] += overlap * nx / 2
                        balls[j]['y'] += overlap * ny / 2
            
            data[seq, t] = np.array(img)
        
        if (seq + 1) % 1000 == 0:
            print(f"  Generated {seq + 1}/{num_sequences}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)
    print(f"Saved to {save_path}")
    print(f"Shape: {data.shape}")
    return data


def generate_pendulum(
    num_sequences=5000,
    seq_length=30,
    img_size=64,
    save_path="data/pendulum.npy"
):
    """
    Generate pendulum motion sequences.
    Simple harmonic motion with damping.
    """
    print(f"Generating {num_sequences} pendulum sequences...")
    
    data = np.zeros((num_sequences, seq_length, img_size, img_size), dtype=np.uint8)
    
    pivot = (img_size // 2, 10)  # Pivot at top center
    length = 25  # Pendulum length
    bob_radius = 5
    
    for seq in range(num_sequences):
        # Random initial angle and angular velocity
        theta = np.random.uniform(-math.pi/3, math.pi/3)
        omega = np.random.uniform(-0.3, 0.3)
        
        g = 0.5  # Gravity
        damping = 0.995
        
        for t in range(seq_length):
            # Create frame
            img = Image.new('L', (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            
            # Bob position
            bob_x = pivot[0] + length * math.sin(theta)
            bob_y = pivot[1] + length * math.cos(theta)
            
            # Draw rod
            draw.line([pivot, (bob_x, bob_y)], fill=128, width=2)
            
            # Draw bob
            draw.ellipse([
                bob_x - bob_radius, bob_y - bob_radius,
                bob_x + bob_radius, bob_y + bob_radius
            ], fill=255)
            
            # Draw pivot
            draw.ellipse([
                pivot[0] - 3, pivot[1] - 3,
                pivot[0] + 3, pivot[1] + 3
            ], fill=180)
            
            data[seq, t] = np.array(img)
            
            # Update physics (simple pendulum equation)
            alpha = -g / length * math.sin(theta)
            omega += alpha
            omega *= damping
            theta += omega
        
        if (seq + 1) % 1000 == 0:
            print(f"  Generated {seq + 1}/{num_sequences}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)
    print(f"Saved to {save_path}")
    print(f"Shape: {data.shape}")
    return data


def generate_spring_mass(
    num_sequences=5000,
    seq_length=30,
    img_size=64,
    save_path="data/spring_mass.npy"
):
    """
    Generate spring-mass system sequences.
    Horizontal oscillation with damping.
    """
    print(f"Generating {num_sequences} spring-mass sequences...")
    
    data = np.zeros((num_sequences, seq_length, img_size, img_size), dtype=np.uint8)
    
    anchor_x = 10
    anchor_y = img_size // 2
    rest_length = 20
    mass_radius = 6
    
    for seq in range(num_sequences):
        # Random initial displacement and velocity
        x = np.random.uniform(rest_length + 5, img_size - mass_radius - 5)
        v = np.random.uniform(-2, 2)
        
        k = 0.3  # Spring constant
        damping = 0.98
        
        for t in range(seq_length):
            # Create frame
            img = Image.new('L', (img_size, img_size), 0)
            draw = ImageDraw.Draw(img)
            
            # Draw spring (zigzag)
            spring_points = [(anchor_x, anchor_y)]
            num_coils = 8
            coil_width = 4
            spring_length = x - anchor_x - mass_radius
            
            for i in range(1, num_coils * 2 + 1):
                px = anchor_x + (i / (num_coils * 2)) * spring_length
                py = anchor_y + (coil_width if i % 2 == 1 else -coil_width)
                spring_points.append((px, py))
            spring_points.append((x - mass_radius, anchor_y))
            
            draw.line(spring_points, fill=128, width=1)
            
            # Draw anchor
            draw.rectangle([anchor_x - 3, anchor_y - 8, anchor_x + 3, anchor_y + 8], fill=100)
            
            # Draw mass
            draw.ellipse([
                x - mass_radius, anchor_y - mass_radius,
                x + mass_radius, anchor_y + mass_radius
            ], fill=255)
            
            data[seq, t] = np.array(img)
            
            # Update physics (Hooke's law)
            displacement = x - anchor_x - rest_length
            a = -k * displacement
            v += a
            v *= damping
            x += v
            
            # Bounds
            x = max(anchor_x + rest_length // 2, min(img_size - mass_radius - 2, x))
        
        if (seq + 1) % 1000 == 0:
            print(f"  Generated {seq + 1}/{num_sequences}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data)
    print(f"Saved to {save_path}")
    print(f"Shape: {data.shape}")
    return data


def visualize_dataset(data_path, save_path, title, num_rows=4, num_cols=10):
    """Visualize samples from a dataset."""
    import matplotlib.pyplot as plt
    
    data = np.load(data_path)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    
    for row in range(num_rows):
        seq_idx = np.random.randint(len(data))
        for col in range(num_cols):
            t = col * (data.shape[1] // num_cols)
            axes[row, col].imshow(data[seq_idx, t], cmap='gray')
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(f't={t}')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['bouncing', 'pendulum', 'spring', 'all'])
    parser.add_argument('--num_sequences', type=int, default=5000)
    parser.add_argument('--seq_length', type=int, default=30)
    args = parser.parse_args()
    
    if args.dataset in ['bouncing', 'all']:
        generate_bouncing_balls_gravity(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length
        )
        visualize_dataset(
            "data/bouncing_balls_gravity.npy",
            "bouncing_balls_preview.png",
            "Bouncing Balls with Gravity"
        )
    
    if args.dataset in ['pendulum', 'all']:
        generate_pendulum(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length
        )
        visualize_dataset(
            "data/pendulum.npy",
            "pendulum_preview.png",
            "Pendulum Motion"
        )
    
    if args.dataset in ['spring', 'all']:
        generate_spring_mass(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length
        )
        visualize_dataset(
            "data/spring_mass.npy",
            "spring_mass_preview.png",
            "Spring-Mass System"
        )
    
    print("\n✅ Dataset generation complete!")
