# import glob
# import contextlib
# from PIL import Image

# # filepaths

# fp_out = "./Phase2_1/lego.gif"

# fp_in = []

# for i in range(200):
#     fp_in.append("./Phase2_1/output/Rendered_Image_"+str(i)+".png")

# # use exit stack to automatically close opened images
# with contextlib.ExitStack() as stack:

#     # lazily load images
#     imgs = (stack.enter_context(Image.open(f))
#             for f in fp_in)

#     # extract  first image from iterator
#     img = next(imgs)

#     # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
#     img.save(fp=fp_out, format='GIF', append_images=imgs,
#              save_all=True, duration=50, loop=0)

import glob
import contextlib
from PIL import Image

# Filepaths
fp_out = "./Phase2_1/output/lego1.gif"

# List of input images
fp_in = ["./Phase2_1/output/Rendered_Image_"+str(i)+".png" for i in range(200)]

# Desired size (width, height)
new_size = (256, 256)  # Example: resize images to 256x256 pixels

# Use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:
    # Lazily load and resize images
    imgs = (stack.enter_context(Image.open(f).resize(new_size, Image.Resampling.LANCZOS))
            for f in fp_in)
    
    # Extract the first image from iterator
    img = next(imgs)
    
    # Save the frames as an animated GIF
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=50, loop=0)
