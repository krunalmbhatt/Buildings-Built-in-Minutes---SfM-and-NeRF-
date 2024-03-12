import torch
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


######################
#### Ray Direction####
######################

def ray_direction(width, height, focal_length):         #Get Ray Direction
    x_coords, y_coords = torch.meshgrid(torch.linspace(0, width - 1, width), 
                                        torch.linspace(0, height - 1, height))
    x = x_coords.t()
    y = y_coords.t()
    directions = torch.stack([(x - width *0.5)/focal_length, -(y - height*0.5)/focal_length, -torch.ones_like(x)], dim=-1)

    # print("directions: ", directions)
    # print("directions.shape: ", directions.shape)
    # print('')
    return directions    

def calculate_rays(pose, directions):                #get Rays
    directions = directions@pose[:, :3].T
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    ray_o = pose[:, 3].expand(directions.shape)
    ray_d = directions.view(-1, 3)
    ray_o = ray_o.view(-1, 3)
    
    return ray_o, ray_d


#######################
## Sample Ray Points ##
#######################


def calc_sample_pts(ray_origin, ray_dir, near, far, n_samples):
    interval = torch.linspace(0.0, 1.0, steps = n_samples, requires_grad=False).to(ray_origin.device)
    n_ray = ray_origin.shape[0]

    interval = interval.unsqueeze(0)
    near = near.unsqueeze(1)
    far = far.unsqueeze(1)

    pt_interval = near + interval * (far - near)

    pertub = torch.rand_like(pt_interval) 
    pertub = (pertub - 0.5) * (far - near) / n_samples
    pertub[:,0] = 0
    pt_interval = pt_interval + pertub
    pt_interval = torch.reshape(pt_interval, (n_ray, -1))

    sample_pts = ray_origin.unsqueeze(1) + ray_dir.unsqueeze(1) * pt_interval.unsqueeze(-1)
    ray_dir = ray_dir.unsqueeze(1).expand(-1, n_samples, -1)

    return sample_pts, ray_dir, pt_interval

