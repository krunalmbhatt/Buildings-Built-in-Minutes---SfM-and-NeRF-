## BUILDINGS BUILT IN MINUTES - PHASE 2
# RBE 549 Computer Vision 
# Under Guidance of Prof. Nitin Sanket

#Coded  by Krunal M Bhatt
# Date: March 7, 2024
# Reference used for coding: https://github.com/murumura/NeRF-Simple/blob/main/src/models/nerf/NeRFSystem.py


import torch
import cv2
import os
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Torchdata
import matplotlib.pyplot as plt

from loader import SyntheticNeRFDataset
from nerfnetwork import vanillaNeRF
from render import step_train, fetch_model
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Train a NeRF model on synthetic data.')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--output_ch', type=int, default=4, help='output channels (default: 4)')
    parser.add_argument('--input_size', type=int, default=3, help='input size (default: 3)')
    parser.add_argument('--width', type=int, default=256, help='network width (default: 256)')
    parser.add_argument('--depth', type=int, default=8, help='network depth (default: 8)')
    return parser.parse_args()

def main():
    
    args = get_args()

    ####################
    ### Hyperparams ####
    ####################

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    output_ch = args.output_ch
    input_size = args.input_size
    width = args.width
    depth = args.depth

    ###########################
    ###### Loading files ######
    ###########################

    path = './nerf_synthetic/lego/transforms_train.json'
    images_path = SyntheticNeRFDataset(path, img_dim= (400,400))
    train_loader = Torchdata.DataLoader(images_path, batch_size = batch_size, shuffle = True)
    n_rays = len(train_loader)

    ############################
    ###### Model and Loss ######
    ############################

    model = vanillaNeRF(depth, width, input_size, output_ch).to(device)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 450000, gamma = 0.1)

    loss_fn = torch.nn.MSELoss()
    epoch = 0

    checkpoint_dir = './Phase2_1/checkpoints'
    checkpoint_filename = 'epoch_{}.pth'.format(epoch)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    writer = SummaryWriter('runs/nerf_training')

    ############################
    ###### Training Loop ###### 
    ############################

    while epoch < 10:
        b_time = time.time()
        condtition = False
        with tqdm(total = n_rays, unit= 'rays') as pbar:
            for k, dat in enumerate(train_loader, 0):
                # print(dat)
                rays = dat['rays'].to(device)
                images = dat['imgs'].to(device)
                # assert not torch.isnan(rays).any(), "NaNs in rays"
                # assert not torch.isnan(images).any(), "NaNs in images"
                renderings = step_train(rays, images, n_samplePoints = 256, model= model)
                # print(renderings)
                rendered_img = renderings['color_map']
                # print("RENSEREDDFSFS", rendered_img)
                loss = loss_fn(rendered_img, images)
                rendered_img.detach()
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                scheduler.step()
                loss = loss.item()
                pbar.update()      

                # Log training loss to TensorBoard
                writer.add_scalar('Training Loss', loss, epoch)


        # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, 
        #            './Phase2_1/checkpoints/epoch_{}.pth'.format(epoch))
        if not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir)
            except OSError as e:
                # If the directory could not be created, use the present working directory
                print("Directory could not be created, saving in the present working directory.")
                checkpoint_path = checkpoint_filename

        # Save the checkpoint
        torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': loss}, checkpoint_path)

        epoch += 1
        print('Epoch: {} Loss: {} Time: {}'.format(epoch, loss, time.time()-b_time))
        delta_t = time.time()-b_time
        tqdm.write('Training ended in {}'.format(delta_t))
    writer.close()

if __name__ == '__main__':
    main()

