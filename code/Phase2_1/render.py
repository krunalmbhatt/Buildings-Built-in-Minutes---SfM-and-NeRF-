import torch
import numpy as np

from nerfnetwork import vanillaNeRF

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    val = torch.cumprod(tensor, dim = -1)
    val = torch.roll(val, 1, dims = -1)
    val[..., 0] = 1.0
    return val

class volumeRender(torch.nn.Module):
    def __init__(self, train_radiance_noise = 0, val_radiance_noise = 0, white_bg = False, att_threshold=1e-3, device='cuda', **kwargs):
        super().__init__()
        self.train_radiance_noise = train_radiance_noise
        self.val_radiance_noise = val_radiance_noise
        self.whitebg = white_bg
        e = torch.tensor([1e10]).to(device)
        e.requires_grad = False
        self.need_grad = False
        self.register_buffer('e', e)
        self.att_threshold = att_threshold

    def forward(self, rad_field, depthVal, ray_dir):
        if self.training:
            radiance_field_std = self.train_radiance_noise
        else:
            radiance_field_std = self.val_radiance_noise

        diff = torch.cat(
            (
                depthVal[..., 1:] - depthVal[..., :-1], 
                self.e.expand(depthVal[...,:1].shape),
            ),
            dim = -1,
        )

        diff = diff * ray_dir[..., None, :].norm(p=2, dim=-1)

        rgb = rad_field[..., :3]

        noise = 0

        if radiance_field_std > 0.0:
            noise = (
                torch.randn(rad_field[..., 3].shape, dtype = rad_field.dtype, device = rad_field.device) * radiance_field_std
            )

        sigma_a = torch.nn.functional.relu(rad_field[..., 3] + noise)

        alpha = 1 - torch.exp(-sigma_a * diff)          #alpha is the accumulated transparency
        T_i = cumprod_exclusive(alpha + self.e)   #torch.cumprod(alpha + self.epsilon, dim = -1), t_i is the accumulated transmittance
        weight = alpha * T_i                           #weight is the accumulated color

        mask = (T_i >  self.att_threshold).float()      #mask is the accumulated mask

        color_map = weight[..., None] * rgb          #color_map is the accumulated color
        color_map = color_map.sum(dim=-2)             #accumulated color

        acc_map = weight.sum(dim=-1)  #accumulated weight

        depth_map = (weight * depthVal).sum(dim=-1)  
        display_map = 1/ torch.max(1e-10*torch.ones_like(depth_map), depth_map/acc_map)  #display map is the accumulated depth map divided by accumulated weight        
        display_map[torch.isnan(display_map)] = 0

        if self.whitebg:
            color_map = color_map + (1.0 - acc_map[..., None])

        out = {'color_map': color_map, 'depth_map': depth_map, 'weight': weight, 'mask': mask, 'acc_map': acc_map, 'display_map': display_map}
        
        return out
    

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

def step_train(ray, images, n_samplePoints, model):
    ray_origin = ray[:,0:3]
    ray_dir = ray[:,3:6]
    renderer = volumeRender()
    near = ray[:,6]
    far = ray[:,7]
    sample_pts, ray_d_exp, pt_interval = calc_sample_pts(ray_origin, ray_dir, near, far, n_samplePoints)
    out = model(sample_pts, ray_d_exp)
    rendered = renderer(out, pt_interval, ray_dir)

    return rendered

def fetch_model():
    return vanillaNeRF()