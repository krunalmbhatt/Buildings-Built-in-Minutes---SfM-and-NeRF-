################ BUILDINGS BUILT IN MINUTES #################
## RBE 549 - Computer Vision
## Under Guidance of Prof. Nitin Sanket


## DATE : March 4, 2024
## WRITTEN BY : Krunal M Bhatt

## This file contains the dataloader for the dataset. It is a simple dataloader that uses the PyTorch's DataLoader


# import os
# import json

# # Print the current working directory to help diagnose path issues
# print("Current working directory:", os.getcwd())

# file_path = './nerf_synthetic/lego/transforms_train.json'

# # Try to open the file
# try:
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     print(data)
# except FileNotFoundError:
#     print(f"File not found: {file_path}. Please check the path and try again.")


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json

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

# Display images and details one by one or for a specific range
for image_id in range(0, 100):  # Adjust range as necessary
    display_image_and_details(data, image_id)
    input("Press Enter to continue to the next image...")  # Pause to control output flow
