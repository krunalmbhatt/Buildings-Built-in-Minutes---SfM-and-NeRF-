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

        sigma = torch.relu(self.sigma(x)).to(sample_pts.device)
        x = self.fc2(x) 
        x_dir = torch.cat((x, sample_dir), -1).to(sample_pts.device)
        x = F.relu(self.direction(x_dir))
        x = F.relu(self.output(x))
        x = torch.cat((x, sigma), -1)
        return x
    

class positionEncoder(nn.Module):
    def __init__(self, input_ch = 3, freq = 10, log_scale = True):
        super(positionEncoder, self).__init__()
        self.freq = freq
        self.input_ch = input_ch
        self.encoder_function = [torch.sin, torch.cos]
        self.output_ch = input_ch * (len(self.encoder_function)*freq+1)
        if log_scale:
            self.freq_bands = 2.0 ** torch.linspace(0.0, freq - 1, freq)
        else:
            self.freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (freq - 1), freq)

    def forward(self, x):
        output = [x]
        for f in self.freq_bands:
            for func in self.encoder_function:
                output.append(func(f * x))

        return torch.cat(output, -1)
    
    def num_out_ch(self):
        return self.output_ch
    
def apply_PE(input_tensor, input_ch=3, freq=10, log_scale=True):
    return positionEncoder(input_ch, freq, log_scale)(input_tensor)