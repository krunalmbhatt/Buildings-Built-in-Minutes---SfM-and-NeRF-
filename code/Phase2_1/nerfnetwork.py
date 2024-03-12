import torch
import torch.nn as nn
import torch.nn.functional as F

class vanillaNeRF(nn.Module):
    def __init__(self, depth=8, width=256, input_size=3, output_ch=4):
        super(vanillaNeRF, self).__init__()
        self.depth = depth
        self.width = width
        self.fc1 = nn.Linear(63, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width + 63, width)
        self.fc4 = nn.Linear(width, 4)
        self.sigma = nn.Linear(width, 1)
        self.direction = nn.Linear(width+27, width//2)
        self.output = nn.Linear(width//2, 3)
        self.positionEncoder = positionEncoder(input_ch = input_size)
        self.directionEncoder = positionEncoder(input_ch = input_size, freq = 4)

    def forward(self, sample_pts, sample_dir):
        sample_pts = self.positionEncoder(sample_pts)
        sample_dir = self.directionEncoder(sample_dir)
        input = sample_pts

        for k in range(self.depth):
            if k == 0:
                x = F.relu(self.fc1(input)).to(sample_pts.device)
            elif k == self.depth//2:
                x = torch.cat([x, input], -1)
                x = F.relu(self.fc3(x))
            else:
                x = F.relu(self.fc2(x))
        
        # print("x after for loop",x)
        sigma = self.sigma(x).to(sample_pts.device)
        
        x = self.fc2(x)
        
        # print("x after fc2", x) 
        
        x_dir = torch.cat([x, sample_dir], -1).to(sample_pts.device)
        
        # print("x_dir", x_dir)
        
        x = F.relu(self.direction(x_dir))

        # print("after dir", x)
        
        x = F.relu(self.output(x))
        
        # print("after output", x)
        
        x = torch.cat([x, sigma], -1)
        
        # print("x from network: ", x)
        # print("X shape: ", x.shape)
        return x
    

class positionEncoder(nn.Module):
    def __init__(self, input_ch: int = 3, freq: int = 10, log_scale: bool = True):
        super(positionEncoder, self).__init__()
        self.freq = freq
        self.input_ch = input_ch
        self.encoder_function = [torch.sin, torch.cos]
        self.output_ch = input_ch * (len(self.encoder_function)*freq+1)
        if log_scale:
            self.freq_band = 2.0 ** torch.linspace(0, freq - 1, freq)
        else:
            self.freq_band = torch.linspace(1, 2**(freq - 1), freq)

    def forward(self, x):
        output = [x]
        for freq in self.freq_band:
            for func in self.encoder_function:
                output.append(func(freq * x))
        return torch.cat(output, -1)
    
    def num_out_ch(self):
        return self.output_ch
    
def apply_PE(input_tensor, input_ch=3, freq=10, log_scale=True):
    encoded_img = positionEncoder(input_ch=input_ch, freq=freq, log_scale=log_scale)
    return encoded_img(input_tensor)