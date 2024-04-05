# Output
<div align="center">
  <img src="/Nerf.gif" alt"NeRF Output">
</div>

# Phase 1: Structure From Motion (SfM)

Run the wrapper.py file in phase 1 folder.

# Phase 2: NeRF 

This project trains a Neural Radiance Field (NeRF) model on synthetic datasets. The implementation leverages PyTorch for model training and evaluation.

## Environment Setup

To run this project, ensure you have Python 3.8+ and pip installed on your system. It's recommended to use a virtual environment:

```bash
python3 -m venv nerf-env
source nerf-env/bin/activate
```

Install the required packages:

```
pip install torch torchvision opencv-python tqdm numpy matplotlib tensorboard
```

## Project Structure

- `train.py`: Main script for training the NeRF model.
- `test.py`: Script for evaluating the trained model.
- `nerfnetwork.py`: Defines the NeRF model architecture.
- `loader.py`: Contains code for loading and preprocessing the dataset.
- `render.py`: Functions for rendering images using the trained NeRF model.
- `utils.py`: Utility functions used across the project.
- `psnr_ssim.py`: Script for calculating PSNR and SSIM metrics.
- `/checkpoints`: Directory to save model checkpoints.
- `/runs`: Directory where TensorBoard logs are saved.
- `/output`: Directory for saving rendered images and other outputs.

## Training

To train the NeRF model, use the following command:

```
python3 train.py --batch_size 1024 --epochs 10 --learning_rate 1e-4 --width 256 --depth 8
```

### Command-Line Arguments

- `--batch_size`: The size of the batch. Default is 1024.
- `--epochs`: Number of training epochs. Default is 10.
- `--learning_rate`: Learning rate for the optimizer. Default is 1e-4.
- `--width`: Width of the layers in the NeRF model. Default is 256.
- `--depth`: Depth (number of layers) of the NeRF model. Default is 8.

## Evaluation

To evalua

```
python test.py
```

Make sur

e to modify `test.py` to load the correct model checkpoint and specify the correct dataset paths.

## Visualization

Training progress and loss can be visualized using TensorBoard:

```
bash
tensorboard --logdir runs
```

Open the provided URL in a web browser to view the training logs.

## Additional Notes

- Ensure the dataset is correctly placed as per the path specified in `loader.py`.
- The `checkpoints` and `output` directories will be created automatically to store model checkpoints and output images, respectively. If not, make the directory in the current working directory.
