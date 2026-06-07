import argparse
import os
import torch
import time
import matplotlib
matplotlib.use('TkAgg')  # Force Tkinter GUI window output
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Utilizing your GPU to compute pi (RAM IS TOO EXPENSIVE!) \n ' \
    'Only for NVIDIA GPUs with CUDA support.')
    parser.add_argument('--output', '-o', help='Output file path', default='pi.txt')
    parser.add_argument('--cleanup', '-c', action='store_true', help='Clean up the  output file after execution')
    args = parser.parse_args()
    global output_path
    output_path = args.output

    result = compute()

    with open(output_path, 'w') as outfile:
        outfile.write(result)

    print(f"Data has been written to '{output_path}'.")

    if args.cleanup:
        cleanup()

def cleanup():
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Cleaned up: '{output_path}' has been removed.")
    else:
        print(f"No cleanup needed: '{output_path}' does not exist.")


def compute():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA is not available. Please run this on a machine with an NVIDIA GPU.")
        return "CUDA not available."
    print(f"Using device: {device}")

    start_time = time.perf_counter()

    total_iterations = 10**9
    chunk_size = 10**7
    num_chunks = total_iterations // chunk_size
    
    total_inside_circle = 0
    
    with torch.no_grad():
        for _ in range(num_chunks):
            # 1. Generate a small, lightweight batch of coordinates
            points = torch.rand(chunk_size, 2, device=device)
            
            # 2. Check which points fall inside the unit circle
            inside_circle = (points[:, 0]**2 + points[:, 1]**2) <= 1
            
            # 3. Sum up the matches and add them to our running total counter
            total_inside_circle += inside_circle.sum().item()
            
            # Memory is instantly freed/reused here when the loop cycles!

    # Calculate final estimation ratio
    pi_estimate = (total_inside_circle / total_iterations) * 4
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Computation completed in {elapsed_time:.2f} seconds.")
    return f"{pi_estimate}"
    

def visualize_monte_carlo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating visualization math on: {device}")
    
    # 1. Use a lightweight number of points for rendering purposes
    num_points = 25000
    
    with torch.no_grad():
        # Generate random x, y coordinates on the GPU
        points = torch.rand(num_points, 2, device=device)
        
        # Calculate distances to see if they are inside the unit circle
        inside_mask = (points[:, 0]**2 + points[:, 1]**2) <= 1
        
        # Estimate Pi just for this small sample batch
        pi_estimate = (inside_mask.sum().item() / num_points) * 4
        
        # 2. Move data back to CPU and convert to NumPy arrays for Matplotlib
        points_cpu = points.cpu().numpy()
        inside_cpu = inside_mask.cpu().numpy()
    
    # 3. Separate points into inside and outside lists for distinct coloring
    inside_points = points_cpu[inside_cpu]
    outside_points = points_cpu[~inside_cpu]
    
    # 4. Initialize the plot
    plt.figure(figsize=(8, 8))
    
    # Plot points inside the circle as blue
    plt.scatter(inside_points[:, 0], inside_points[:, 1], color='#3498db', s=1, label='Inside Circle')
    # Plot points outside the circle as red
    plt.scatter(outside_points[:, 0], outside_points[:, 1], color='#e74c3c', s=1, label='Outside Circle')
    
    # 5. Draw the actual mathematical circle boundary for reference
    circle_theta = np.linspace(0, np.pi/2, 100)
    plt.plot(np.cos(circle_theta), np.sin(circle_theta), color='black', linewidth=2, label='True Circle Border')
    
    # Style the plot beautifully
    plt.title(f"Monte Carlo Pi Estimation\nPoints: {num_points:,} | Estimated $\\pi \\approx$ {pi_estimate:.4f}", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='lower left', markerscale=5)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Show the plot window!
    print("Opening plot window...")
    plt.show()

if __name__ == "__main__":
    main()
    visualize_monte_carlo()
