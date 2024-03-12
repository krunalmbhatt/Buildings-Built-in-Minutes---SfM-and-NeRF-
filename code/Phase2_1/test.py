
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Torchdata
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision
from torchvision import transforms as T

from loader import SyntheticNeRFDataset
from nerfnetwork import vanillaNeRF
from render import step_train, fetch_model
from tqdm import tqdm
import time
from PIL import Image
import glob
import contextlib

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def project_to_depth(depth, cmap= cv2.COLORMAP_JET):
    a = depth.cpu().numpy()
    a = np.nan_to_num(a)     #change to 0
    minimum = np.min(a)
    maximum = np.max(a) 
    b = a - minimum
    c = maximum - minimum + 1e-8
    a = b/c
    a = (255 *a).astype(np.uint8)
    a_ = Image.fromarray(cv2.applyColorMap(a, cmap))
    a_ = T.ToTensor()(a_) 
    return a_

def save_output_images(im, output):
    im = im / 2 + 0.5
    im = torchvision.utils.make_grid(im)
    torchvision.utils.save_image(im.clone(), output, nrow=8)


def main():
    ####################
    ### Hyperparams ####
    ####################

    # batch_size = 32
    batch_size = 1000
    epochs = 100
    learning_rate = 1e-4
    n_samples = 1000
    output_ch = 4
    input_size = 3
    width = 256
    depth = 8

    print('Hyperparams Done')
    
    ###########################
    ###### Loading model ######
    ###########################

    model = vanillaNeRF(depth,width,input_size, output_ch).to(device)
    ckpt_path = './Phase2_1/checkpoints/epoch_0.pth'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print('checkpoint loaded')


    ##########################
    #### Loading dataset #####
    ##########################

    test_path = './nerf_synthetic/lego/transforms_test.json'
    test_dataset = SyntheticNeRFDataset(test_path, img_dim=(400,400))
    test_loader = Torchdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n_rays = len(test_loader)

    print('Test_loaded')


    ##########################
    ###### Testing Loop ######
    ##########################

    image_bool = False
    output_img = []
    gen_imgs = []
    for i, data in enumerate(test_loader, 0):
        rays = data['rays'].to(device)
        images = data['imgs'].to(device)
        postprocess = step_train(rays, images, n_samplePoints=32, model=model)
        rendered_img = postprocess['color_map']
        output_img.append(rendered_img.cpu().detach())
        if(i+1)%160 == 0 and i != 0:
            gen_imgs.append(output_img)
            output_img = []
        
        print(i)

    print('halfway there')
    i = 0
    for g in gen_imgs:
        for img in g:
            if image_bool==False:
                rendered_op = img
                image_bool = True
            else:
                rendered_op = torch.cat([rendered_op, img], dim =0)

        print(rendered_op.shape)
        img = rendered_op.view(400,400,3).cpu()
        img_ = img.permute(2,0,1)
        # save_img(img_,'./output/Rendered_Image_'+str(i)+'.png')

        figure, a = plt.subplots()
        a.imshow(img, aspect='auto')
        a.axis('off')

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        a.xaxis.set_major_locator(plt.NullLocator())
        a.yaxis.set_major_locator(plt.NullLocator())

        plt.savefig('./Phase2_1/output/Rendered_Image_'+str(i)+'.png', bbox_inches='tight', pad_inches=0, dpi=300)


        i +=1
        image_bool = False
        
    print('Complete')
    pass


    ##########################
    ###### Saving GIF #######
    ##########################

    gif_out = './Phase2_1/output/lego.gif'
    gif_in = []

    for i in range(200):
        gif_in.append('./Phase2_1/output/Rendered_Image_'+str(i)+'.png')

    with contextlib.ExitStack() as STACK:

        images = (STACK.enter_context(Image.open(f))
               for f in gif_in)
        
        img = next(images)

        img.save(gif_out, format='GIF', append_images= images, save_all = True, duration = 50, loop =0)


if __name__ == '__main__':
    main()