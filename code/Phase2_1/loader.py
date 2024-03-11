import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


####################
#### Load JSON #####
####################

def loadJson(path):
    with open(path) as file:
        data = json.load(file)
    return data

def frames_from_data(frames):                       # Get Frame Data
    matrices = np.array(frames['transform_matrix'])   
    pose = matrices[:3, :4]
    image_data = './nerf_synthetic/lego'        #path string
    # image_data = './nerf_synthetic/ship/train/'        
    image_path = frames['file_path']
    image_path = image_path.replace('.','')
    image_data = image_data + image_path + '.png'
    img = Image.open(image_data)
    img  = img.resize((400, 400), Image.LANCZOS)  
    # cv2.imshow('image',np.array(img))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return pose, img

######################
#### Ray Direction####
######################

def ray_direction(width, height, focal_length):         #Get Ray Direction
    x_coords, y_coords = torch.meshgrid(torch.linspace(0, width - 1, width), 
                                        torch.linspace(0, height - 1, height))
    x = x_coords.t()
    y = y_coords.t()
    directions = torch.stack([(x - width *0.5)/focal_length, -(y - height*0.5)/focal_length, -torch.ones_like(x)], dim=-1)

    print("directions: ", directions)
    print("directions.shape: ", directions.shape)
    print('')
    return directions    

def calculate_rays(pose, directions):                #get Rays
    directions = directions@pose[:, :3].T
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    ray_o = pose[:, 3].expand(directions.shape)
    ray_d = directions.view(-1, 3)
    ray_o = ray_o.view(-1, 3)
    
    return ray_o, ray_d


######################
#### NeRF Dataset ####
######################

class SyntheticNeRFDataset(Dataset):
    def __init__(self, path, img_dim=(400, 400)):
        self.path = path
        self.width, self.height = img_dim
        self.transforms = transforms.ToTensor()
        self.get_data()

    def get_data(self):
        self.data = loadJson(self.path)
        focal_length = 0.5 * self.width / np.tan(0.5 * self.data['camera_angle_x'])  #From : https://github.com/yenchenlin/nerf-pytorch/issues/41
        self.far = 6.0
        self.near = 2.0
        self.range = torch.tensor([self.near, self.far])
        self.directions = ray_direction(self.width, self.height, focal_length)
        self.rays = []
        self.images = []
        for frame in self.data['frames']:
            pose, img = frames_from_data(frame)
            img = self.transforms(img)
            pose = torch.FloatTensor(pose)
            img = img.view(4, -1).permute(1,0)
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
            self.images.append(img)
            ray_o, ray_d = calculate_rays(pose, self.directions)
            self.rays.append(torch.cat([ray_o, ray_d, self.range[0]*torch.ones_like(ray_o[:, :1]), self.range[1]*torch.ones_like(ray_o[:, :1])], 1))
        
        self.rays = torch.cat(self.rays, dim=0)
        self.images = torch.cat(self.images, dim=0)

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, idx):
        return {'rays': self.rays[idx], 'imgs': self.images[idx]}
    

## The code in this file is used to load the dataset and prepare the rays and images for training. The dataset is loaded from a 
# JSON file containing the metadata of the dataset. The rays are calculated using the camera pose and the focal length of the camera. 
# The images are resized to the specified dimensions and converted to tensors. 
# The rays and images are then stored in the dataset buffers for training.
# The dataset class is used to create a PyTorch dataset that can be used with the DataLoader class for training the NeRF model.

## The SyntheticNeRFDataset class is used to load the dataset and prepare the rays and images for training.
# The class takes the path to the JSON file containing the metadata of the dataset and the dimensions of the images as input. 
# The get_data method is used to load the dataset from the JSON file and prepare the rays and images for training. 
# The method calculates the focal length of the camera and the directions of the rays through each pixel in the camera frame.
# It then iterates through the frames in the metadata and calculates the rays and images for each frame. 
# The rays and images are stored in the rays and images buffers, which are then used to create a PyTorch dataset.

## The __len__ method returns the number of rays in the dataset, and the __getitem__ method returns a dictionary containing the rays 
# and images for a given index. The rays and images are then used to train the NeRF model using the DataLoader class.

## The loadJson function is used to load the metadata from the JSON file, and the frames_from_data function is used to extract the
# camera pose and image data from the metadata. The ray_direction function is used to calculate the directions of the rays through 
# each pixel in the camera frame, and the calculate_rays function is used to transform the rays to world coordinates using the camera
# pose. The SyntheticNeRFDataset class is then used to create a PyTorch dataset that can be used with the DataLoader class for 
# training the NeRF model.
    
