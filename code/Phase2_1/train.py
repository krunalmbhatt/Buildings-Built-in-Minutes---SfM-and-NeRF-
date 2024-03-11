import torch
import cv2

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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main():
    ####################
    ### Hyperparams ####
    ####################

    # batch_size = 32
    batch_size = 600
    epochs = 100
    learning_rate = 1e-4
    n_samples = 1000
    output_ch = 4
    input_size = 3
    width = 256
    depth = 8

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 45000, gamma = 0.1)

    loss_fn = torch.nn.MSELoss()
    epoch = 0

    ############################
    ###### Training Loop ######
    ############################

    while epoch < epochs:
        b_time = time.time()
        condtition = False
        with tqdm(total = n_rays, unit= 'rays') as pbar:
            for k, dat in enumerate(train_loader, 0):
                # print(dat)
                rays = dat['rays'].to(device)
                images = dat['imgs'].to(device)
                renderings = step_train(rays, images, n_samplePoints = 256, model= model)
                # print(renderings)
                rendered_img = renderings['color_map']

                loss = loss_fn(rendered_img, images)
                rendered_img.detach()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss = loss.item()
                pbar.update()
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, 
                   './checkpoints/epoch_{}.pth'.format(epoch))
        epoch += 1
        print('Epoch: {} Loss: {} Time: {}'.format(epoch, loss, time.time()-b_time))
        delta_t = time.time()-b_time
        tqdm.write('Training ended in {}'.format(delta_t))


        ############################
        ####### Testing Loop #######
        ############################

        test_path = './nerf_synthetic/lego/transforms_test.json'
        test_dataset = SyntheticNeRFDataset()




if __name__ == '__main__':
    main()