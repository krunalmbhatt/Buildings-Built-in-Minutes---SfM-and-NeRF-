import torch
from torch import nn
from typing import Tuple
from typing import Optional
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
  r"""
  Sine-cosine positional encoder for input points.
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # Define frequencies in either linear or log scale
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # Alternate sin and cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(
    self,
    x
  ) -> torch.Tensor:
    
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
  


class NeRF(nn.Module):
   def __init__(
      self,
      d_input: int = 3,
      n_layers: int = 8,
      d_filter: int = 256,
      skip: Tuple[int] = (4,),
      d_viewdirs: Optional[int] = None):
      
      super().__init__()
      self.d_input = d_input
      self.skip = skip
      self.act = nn.functional.relu
      self.d_viewdirs = d_viewdirs
      
      # Create model layers
      self.layers = nn.ModuleList(
        [nn.Linear(self.d_input, d_filter)] +
        [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
        else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )
      
      # Bottleneck layers
      if self.d_viewdirs is not None:
        # If using viewdirs, split alpha and RGB
        self.alpha_out = nn.Linear(d_filter, 1)
        self.rgb_filters = nn.Linear(d_filter, d_filter)
        self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
        self.output = nn.Linear(d_filter // 2, 3)
      else:
        # If no viewdirs, use simpler output
        self.output = nn.Linear(d_filter, 4)
        
   def forward(self, x: torch.Tensor, viewdirs: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    # Cannot use viewdirs if instantiated with d_viewdirs = None
      if self.d_viewdirs is None and viewdirs is not None:
        raise ValueError('Cannot input x_direction if d_viewdirs was not given.')
      
    # Apply forward pass up to bottleneck
      x_input = x
      for i, layer in enumerate(self.layers):
        x = self.act(layer(x))
        if i in self.skip:
            x = torch.cat([x, x_input], dim=-1)

    # Apply bottleneck
      if self.d_viewdirs is not None:
        # Split alpha from network output
        alpha = self.alpha_out(x)
        # Pass through bottleneck to get RGB
        x = self.rgb_filters(x)
        x = torch.concat([x, viewdirs], dim=-1)
        x = self.act(self.branch(x))
        x = self.output(x)
        #Concatenate alphas to output
        x = torch.concat([x, alpha], dim=-1)
      else:
          # Simple output
          x = self.output(x)
      return x


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
   r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)
    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
    is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
    tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
   cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
   cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
   cumprod[..., 0] = 1.
   
   return cumprod


def outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert the raw NeRF output into RGB and other maps.
    """
    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    
    # Multiply each distance by the norm of its corresponding direction ray to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1) # [n_rays, n_samples]
    
    # Add noise to model's predictions for density. Can be used to regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
       noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    
    # Predict density of each sample along each ray. Higher values imply higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]. The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)
    
    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]
    
    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    #Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    
    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)
    
    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
       rgb_map = rgb_map + (1. - acc_map[..., None])
    
    
    return rgb_map, depth_map, acc_map, weights
