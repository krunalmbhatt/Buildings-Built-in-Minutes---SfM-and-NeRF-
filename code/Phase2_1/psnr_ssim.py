import cv2
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np

def calculate_metrics(rendered_images_dir, test_images_dir):
    rendered_images = os.listdir(rendered_images_dir)
    psnr_values = []
    ssim_values = []

    for image_name in rendered_images:
        rendered_image_path = os.path.join(rendered_images_dir, image_name)
        test_image_path = os.path.join(test_images_dir, image_name)

        # Load images
        rendered_image = cv2.imread(rendered_image_path)
        test_image = cv2.imread(test_image_path)

        # Convert images to grayscale for SSIM calculation
        rendered_image_gray = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2GRAY)
        test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # Calculate PSNR
        psnr = cv2.PSNR(rendered_image, test_image)
        psnr_values.append(psnr)

        # Calculate SSIM
        ssim_value = ssim(rendered_image_gray, test_image_gray)
        ssim_values.append(ssim_value)

        print(f"Image: {image_name}, PSNR: {psnr}, SSIM: {ssim_value}")

    # Calculate average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")

# Example usage
rendered_images_dir = './Phase2_1/output'
test_images_dir = './nerf_synthetic/lego/test'
calculate_metrics(rendered_images_dir, test_images_dir)
