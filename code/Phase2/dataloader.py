################ BUILDINGS BUILT IN MINUTES #################
## RBE 549 - Computer Vision
## Under Guidance of Prof. Nitin Sanket


## DATE : March 4, 2024
## WRITTEN BY : Krunal M Bhatt

## This file contains the dataloader for the dataset. It is a simple dataloader that uses the PyTorch's DataLoader


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import numpy as np
import torch

#######################################################
#################LOAD THE JSON FILE####################
#######################################################
# Load JSON data\
def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


# Function to display an image and its details
def display_image_and_details(data, image_id):
    # Formulate file_path suffix
    image_suffix = f'r_{image_id}'
    
    # Initialize variables to store rotation and transform_matrix
    rotation = None
    transform_matrix = None

    # Search for the specified image and its properties
    for frame in data['frames']:
        if frame['file_path'].endswith(image_suffix):
            rotation = frame['rotation']
            transform_matrix = frame['transform_matrix']
            break

    # Check if the image and its data were found
    if rotation is not None and transform_matrix is not None:
        # Display the image
        img_path = f'./nerf_synthetic/lego/train/r_{image_id}.png'  # Update this path/format as necessary
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(f"Image: r_{image_id}")
        plt.axis('off')  # Hide axis
        plt.show()

        # Print rotation and transformation matrix
        print(f"Rotation: {rotation}")
        print("Transformation Matrix:")
        for row in transform_matrix:
            print(row)
    else:
        print(f"Image data for r_{image_id} not found.")

# Example usage
json_file_path = './nerf_synthetic/lego/transforms_train.json'  # Update this to your JSON file's path
data = load_data(json_file_path)

# # Display images and details one by one or for a specific range
# for image_id in range(0, 100):  # Adjust range as necessary
#     display_image_and_details(data, image_id)
#     input("Press Enter to continue to the next image...")  # Pause to control output flow



#######################################################
###############PLOT THE GIVEN POSES####################
#######################################################

# Corrected approach for extracting transformation matrices
def extract_transformations_corrected(data):
    transform_matrices = [np.array(frame['transform_matrix']) for frame in data['frames']]
    origins = np.array([mat[:3, 3] for mat in transform_matrices])
    directions = - np.array([mat[:3, 2] for mat in transform_matrices])  # Camera looking direction in -Z
    return origins, directions

# Plot the directions and origins in a 3D quiver plot
def plot_3d_quiver(origins, directions):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
              directions[:, 0], directions[:, 1], directions[:, 2],
              length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Camera Positions and Directions')
    plt.show()

# Extract and plot using the corrected function
origins_corrected, directions_corrected = extract_transformations_corrected(data)
plot_3d_quiver(origins_corrected, directions_corrected)


#######################################################
###############COMPUTE RAYS USING PYTORCH##############
#######################################################

def compute_rays_torch(height, width, focal_length, transform_matrix):
    """
    Compute the origins and directions of rays for each pixel in an image using PyTorch.

    Args:
    - height: The height of the image, as an int.
    - width: The width of the image, as an int.
    - focal_length: The focal length of the camera, as a float or tensor.
    - transform_matrix: The camera-to-world transformation matrix, as a tensor of shape [4, 4].

    Returns:
    - ray_origins: The origins of the rays in world coordinates, as a tensor of shape [height, width, 3].
    - ray_directions: The normalized directions of the rays in world coordinates, as a tensor of shape [height, width, 3].
    """
    # Meshgrid for pixel coordinates
    i, j = torch.meshgrid(torch.arange(width, dtype=torch.float32), 
                          torch.arange(height, dtype=torch.float32), indexing='xy')
    # Convert to normalized device coordinates
    x = (i - width / 2) / focal_length
    y = -(j - height / 2) / focal_length
    z = -torch.ones_like(x)

    # Directions in camera coordinates
    directions_camera = torch.stack([x, y, z], dim=-1)
    directions_camera /= torch.norm(directions_camera, dim=-1, keepdim=True)

    # Transform directions to world space
    ray_directions = torch.einsum('hwk,kl->hwl', directions_camera, transform_matrix[:3, :3])

    # Ray origins: camera position in world space
    ray_origins = transform_matrix[:3, 3].expand(height, width, 3)

    return ray_origins, ray_directions
