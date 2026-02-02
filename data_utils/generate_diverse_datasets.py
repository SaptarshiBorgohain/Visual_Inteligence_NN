"""
Generate Diverse Physics Datasets for World Model Training
- Multiple object interactions
- Different physics phenomena
- More complex scenarios
"""
import numpy as np
import os
from tqdm import tqdm

def generate_multi_ball_collision(n_samples=3000, n_frames=30, size=64):
    """Multiple balls with collisions and gravity."""
    print("Generating multi-ball collision dataset...")
    data = np.zeros((n_samples, n_frames, size, size), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        n_balls = np.random.randint(2, 5)  # 2-4 balls
        
        # Ball properties
        balls = []
        for _ in range(n_balls):
            balls.append({
                'x': np.random.uniform(10, size-10),
                'y': np.random.uniform(10, size-30),
                'vx': np.random.uniform(-3, 3),
                'vy': np.random.uniform(-1, 2),
                'r': np.random.randint(3, 7),
            })
        
        gravity = 0.15
        
        for t in range(n_frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            
            # Update each ball
            for b in balls:
                b['vy'] += gravity
                b['x'] += b['vx']
                b['y'] += b['vy']
                
                # Wall collisions
                if b['x'] - b['r'] < 0:
                    b['x'] = b['r']
                    b['vx'] *= -0.9
                if b['x'] + b['r'] >= size:
                    b['x'] = size - b['r'] - 1
                    b['vx'] *= -0.9
                if b['y'] + b['r'] >= size:
                    b['y'] = size - b['r'] - 1
                    b['vy'] *= -0.8
                if b['y'] - b['r'] < 0:
                    b['y'] = b['r']
                    b['vy'] *= -0.9
            
            # Ball-ball collisions
            for j, b1 in enumerate(balls):
                for k, b2 in enumerate(balls[j+1:], j+1):
                    dx = b2['x'] - b1['x']
                    dy = b2['y'] - b1['y']
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = b1['r'] + b2['r']
                    
                    if dist < min_dist and dist > 0:
                        # Elastic collision
                        nx, ny = dx/dist, dy/dist
                        dvx = b1['vx'] - b2['vx']
                        dvy = b1['vy'] - b2['vy']
                        dvn = dvx*nx + dvy*ny
                        
                        b1['vx'] -= dvn * nx
                        b1['vy'] -= dvn * ny
                        b2['vx'] += dvn * nx
                        b2['vy'] += dvn * ny
                        
                        # Separate balls
                        overlap = min_dist - dist
                        b1['x'] -= overlap/2 * nx
                        b1['y'] -= overlap/2 * ny
                        b2['x'] += overlap/2 * nx
                        b2['y'] += overlap/2 * ny
            
            # Draw balls
            for b in balls:
                y, x = np.ogrid[:size, :size]
                mask = (x - b['x'])**2 + (y - b['y'])**2 <= b['r']**2
                frame[mask] = 255
            
            data[i, t] = frame
    
    return data


def generate_projectile_motion(n_samples=3000, n_frames=30, size=64):
    """Projectile motion with different angles and velocities."""
    print("Generating projectile motion dataset...")
    data = np.zeros((n_samples, n_frames, size, size), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        # Launch parameters
        x = np.random.uniform(5, 15)
        y = size - 10
        angle = np.random.uniform(30, 75) * np.pi / 180
        speed = np.random.uniform(4, 8)
        vx = speed * np.cos(angle)
        vy = -speed * np.sin(angle)
        
        gravity = 0.2
        radius = np.random.randint(3, 6)
        
        for t in range(n_frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            
            vy += gravity
            x += vx
            y += vy
            
            # Ground bounce
            if y + radius >= size:
                y = size - radius - 1
                vy *= -0.7
                vx *= 0.95
            
            # Walls
            if x - radius < 0 or x + radius >= size:
                vx *= -0.8
                x = np.clip(x, radius, size - radius - 1)
            
            # Draw
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - x)**2 + (yy - y)**2 <= radius**2
            frame[mask] = 255
            
            data[i, t] = frame
    
    return data


def generate_double_pendulum(n_samples=2000, n_frames=30, size=64):
    """Chaotic double pendulum motion."""
    print("Generating double pendulum dataset...")
    data = np.zeros((n_samples, n_frames, size, size), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        # Pendulum parameters
        L1, L2 = 15, 12
        m1, m2 = 1.0, 1.0
        g = 0.5
        
        theta1 = np.random.uniform(-np.pi/2, np.pi/2)
        theta2 = np.random.uniform(-np.pi/2, np.pi/2)
        omega1 = np.random.uniform(-0.5, 0.5)
        omega2 = np.random.uniform(-0.5, 0.5)
        
        cx, cy = size//2, 15
        dt = 0.3
        
        for t in range(n_frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            
            # Double pendulum physics
            delta = theta2 - theta1
            den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
            den2 = (L2 / L1) * den1
            
            alpha1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                     m2 * g * np.sin(theta2) * np.cos(delta) +
                     m2 * L2 * omega2**2 * np.sin(delta) -
                     (m1 + m2) * g * np.sin(theta1)) / den1
            
            alpha2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                     (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                     (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                     (m1 + m2) * g * np.sin(theta2)) / den2
            
            omega1 += alpha1 * dt
            omega2 += alpha2 * dt
            theta1 += omega1 * dt
            theta2 += omega2 * dt
            
            # Damping
            omega1 *= 0.999
            omega2 *= 0.999
            
            # Calculate positions
            x1 = cx + L1 * np.sin(theta1)
            y1 = cy + L1 * np.cos(theta1)
            x2 = x1 + L2 * np.sin(theta2)
            y2 = y1 + L2 * np.cos(theta2)
            
            # Draw pendulum
            # Rod 1
            for j in range(int(L1)):
                px = int(cx + j * np.sin(theta1))
                py = int(cy + j * np.cos(theta1))
                if 0 <= px < size and 0 <= py < size:
                    frame[py, px] = 128
            
            # Rod 2
            for j in range(int(L2)):
                px = int(x1 + j * np.sin(theta2))
                py = int(y1 + j * np.cos(theta2))
                if 0 <= px < size and 0 <= py < size:
                    frame[py, px] = 128
            
            # Bob 1
            yy, xx = np.ogrid[:size, :size]
            mask1 = (xx - x1)**2 + (yy - y1)**2 <= 16
            frame[mask1] = 200
            
            # Bob 2
            mask2 = (xx - x2)**2 + (yy - y2)**2 <= 16
            frame[mask2] = 255
            
            data[i, t] = frame
    
    return data


def generate_falling_objects(n_samples=3000, n_frames=30, size=64):
    """Different shaped objects falling with air resistance."""
    print("Generating falling objects dataset...")
    data = np.zeros((n_samples, n_frames, size, size), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        # Object type: 0=circle, 1=square, 2=triangle
        obj_type = np.random.randint(0, 3)
        x = np.random.uniform(15, size-15)
        y = np.random.uniform(5, 15)
        vx = np.random.uniform(-1, 1)
        vy = 0
        obj_size = np.random.randint(5, 10)
        
        gravity = 0.25
        drag = 0.02 if obj_type == 0 else 0.03 if obj_type == 1 else 0.04
        
        for t in range(n_frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            
            # Physics
            vy += gravity
            vy -= drag * vy * abs(vy)  # Air resistance
            vx -= drag * vx * abs(vx)
            x += vx
            y += vy
            
            # Ground
            if y + obj_size >= size:
                y = size - obj_size - 1
                vy *= -0.5
            
            # Walls
            if x - obj_size < 0 or x + obj_size >= size:
                vx *= -0.8
                x = np.clip(x, obj_size, size - obj_size - 1)
            
            # Draw object
            yy, xx = np.ogrid[:size, :size]
            
            if obj_type == 0:  # Circle
                mask = (xx - x)**2 + (yy - y)**2 <= obj_size**2
            elif obj_type == 1:  # Square
                mask = (abs(xx - x) <= obj_size) & (abs(yy - y) <= obj_size)
            else:  # Triangle (approximate)
                mask = ((yy >= y) & (yy <= y + obj_size) & 
                       (abs(xx - x) <= (y + obj_size - yy) * obj_size / obj_size))
            
            frame[mask] = 255
            data[i, t] = frame
    
    return data


def generate_wave_motion(n_samples=2000, n_frames=30, size=64):
    """Wave propagation and interference."""
    print("Generating wave motion dataset...")
    data = np.zeros((n_samples, n_frames, size, size), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        n_sources = np.random.randint(1, 3)
        sources = []
        for _ in range(n_sources):
            sources.append({
                'x': np.random.uniform(10, size-10),
                'y': np.random.uniform(10, size-10),
                'freq': np.random.uniform(0.3, 0.6),
                'phase': np.random.uniform(0, 2*np.pi),
            })
        
        wavelength = np.random.uniform(8, 15)
        
        for t in range(n_frames):
            frame = np.zeros((size, size), dtype=np.float32)
            
            yy, xx = np.ogrid[:size, :size]
            
            for s in sources:
                dist = np.sqrt((xx - s['x'])**2 + (yy - s['y'])**2)
                wave = np.sin(2 * np.pi * (dist / wavelength - s['freq'] * t) + s['phase'])
                amplitude = 1.0 / (1 + 0.05 * dist)  # Decay with distance
                frame += wave * amplitude
            
            # Normalize to 0-255
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
            data[i, t] = (frame * 255).astype(np.uint8)
    
    return data


def generate_orbital_motion(n_samples=2000, n_frames=30, size=64):
    """Orbital mechanics - planets around a star."""
    print("Generating orbital motion dataset...")
    data = np.zeros((n_samples, n_frames, size, size), dtype=np.uint8)
    
    for i in tqdm(range(n_samples)):
        cx, cy = size//2, size//2
        
        n_planets = np.random.randint(1, 4)
        planets = []
        for _ in range(n_planets):
            r = np.random.uniform(10, 25)
            angle = np.random.uniform(0, 2*np.pi)
            speed = np.random.uniform(0.08, 0.15) / np.sqrt(r/15)  # Kepler's law
            planets.append({
                'r': r,
                'angle': angle,
                'speed': speed,
                'size': np.random.randint(2, 5),
            })
        
        for t in range(n_frames):
            frame = np.zeros((size, size), dtype=np.uint8)
            
            # Draw star
            yy, xx = np.ogrid[:size, :size]
            star_mask = (xx - cx)**2 + (yy - cy)**2 <= 25
            frame[star_mask] = 200
            
            # Draw planets
            for p in planets:
                p['angle'] += p['speed']
                px = cx + p['r'] * np.cos(p['angle'])
                py = cy + p['r'] * np.sin(p['angle'])
                
                planet_mask = (xx - px)**2 + (yy - py)**2 <= p['size']**2
                frame[planet_mask] = 255
            
            data[i, t] = frame
    
    return data


def main():
    os.makedirs('data', exist_ok=True)
    
    print("="*60)
    print("Generating Diverse Physics Datasets")
    print("="*60)
    
    # Generate all datasets
    datasets = {
        'multi_ball_collision': generate_multi_ball_collision,
        'projectile_motion': generate_projectile_motion,
        'double_pendulum': generate_double_pendulum,
        'falling_objects': generate_falling_objects,
        'wave_motion': generate_wave_motion,
        'orbital_motion': generate_orbital_motion,
    }
    
    for name, gen_func in datasets.items():
        filepath = f'data/{name}.npy'
        if os.path.exists(filepath):
            print(f"  {name} already exists, skipping...")
            continue
        
        data = gen_func()
        np.save(filepath, data)
        print(f"  Saved {filepath}: {data.shape}")
    
    print("\n" + "="*60)
    print("✅ All datasets generated!")
    print("="*60)
    
    # Summary
    print("\nDataset Summary:")
    total = 0
    for name in datasets.keys():
        filepath = f'data/{name}.npy'
        if os.path.exists(filepath):
            d = np.load(filepath)
            print(f"  {name}: {d.shape[0]} sequences")
            total += d.shape[0]
    print(f"\nTotal: {total} sequences")


if __name__ == "__main__":
    main()
