import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from network import *
from dataloader import *
import os


json_path = './nerf_synthetic/lego/transforms_train.json'
dataset_path = './nerf_synthetic/lego/train/'
num_epochs = 100
batch_size = 5
learning_rate = 1e-4

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def LegoDataset(dataset_path):
    dataset = (dataset_path, 'r_*.png')
    return dataset

def main():
    # Create model
    model = NeRF()
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load dataset
    dataset = LegoDataset(dataset_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, poses) in enumerate(train_loader):
            images = images.to(device)
            poses = poses.to(device)

            # Forward pass
            outputs = model(poses)
            loss = criterion(outputs, images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    main()
