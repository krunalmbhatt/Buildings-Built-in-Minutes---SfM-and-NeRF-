import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import numpy as np
import torch
import os
from dataloader import *
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

height = 800
width = 800
data = load_data(json_file_path)
_, focal_length = get_pose_and_focal_length(data,width)

transform_matrices = extract_transform_matrices(data)

with torch.no_grad():
    ray_origin, ray_direction = compute_rays_torch(height, width, focal_length, transform_matrices)

def sample_stratified_batch(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = False,
    inverse_depth: bool = False
):
    device = rays_o.device
    batch_size, height, width, _ = rays_o.shape

    # Linear space or inverse depth
    if inverse_depth:
        z_vals = 1.0 / (torch.linspace(1.0 / near, 1.0 / far, n_samples, device=device))
    else:
        z_vals = torch.linspace(near, far, n_samples, device=device)

    z_vals = z_vals.expand(batch_size, height, width, n_samples)

    if perturb:
        # Get intervals between samples
        z_intervals = z_vals[..., 1:] - z_vals[..., :-1]
        # Random values for each interval
        random_offsets = torch.rand_like(z_intervals)
        # Perturb z_vals by a fraction of the interval size
        z_vals[..., :-1] += random_offsets * z_intervals
        z_vals[..., -1] = far  # Ensure last value is always 'far'

    # Broadcast ray origins and directions to match the number of samples
    rays_o = rays_o.unsqueeze(-2).expand(-1, -1, -1, n_samples, -1)
    rays_d = rays_d.unsqueeze(-2).expand(-1, -1, -1, n_samples, -1)

    # Compute 3D points: BxHxWxSx3
    pts = rays_o + rays_d * z_vals[..., None]

    return pts, z_vals

# Example usage assuming ray_origin and ray_direction are [B, H, W, 3] tensors
# where B is the number of images, H and W are the image dimensions
near = 2.0
far = 6.0
n_samples = 5
pts, z_vals = sample_stratified_batch(
    ray_origin,
    ray_direction,
    near,
    far,
    n_samples,
    perturb=False,
    inverse_depth=False
)

print(f"Sampled Points Shape: {pts.shape}")
print(f"Depth Values Shape: {z_vals.shape}")

print('Input Points')
print(pts.shape)
print('')
print('Distances Along Ray')
print(z_vals.shape)


# Select a random image from the batch
image_index = 'r_0'
file_path = './nerf_synthetic/lego/train/'
image = os.path.join(file_path, image_index + '.png')

def visualize_sampled_points(pts, sample_index=0):
    sample_pts = pts[sample_index].cpu().numpy()
    sample_pts = sample_pts.reshape(-1, 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_pts[:, 0], sample_pts[:, 1], sample_pts[:, 2], s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Visualizing Sampled Points Along Rays')
    plt.show()

# visualize_sampled_points(pts, sample_index=0)