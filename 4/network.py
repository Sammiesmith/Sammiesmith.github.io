import torch
import torch.nn as nn
import numpy as np
from positional_encoding import positional_encoding

# define the model architecture
class NeuralField2D(nn.Module): # use in built in starter class
    """
    MLP network for representing 2D images as a neural field.
    workflow: positionally encoded coords --> linear (256) --> ReLU --> linear (256) --> ReLU --> linear (256) --> ReLU --> linear (3) --> sigmoid --> RGB colors
    """

    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = input tensor w shape (batch size, input_dim). Assume they are already positionally encoded
        # return output tensor w shape (batch size, 3) of RGB colors in [0,1]
        return self.model(x)
    

class NeRF(nn.Module):
    """
    neural radiance field for 3d scene representation. 
    inputs: 
    - 3D position (x,y,z) that has been positionally encoded
    - viewing direction (theta, pi) that has been positionally encoded
    output: RGB color + density sigma
    architecture from the proj spec website
    """
    def __init__(self, pos_enc_dim=63, dir_enc_dim=27):
        # world_coord_3D_PE_dims = dimension of positinally encoded 3D coords
        # for L=10: 3 * (1 + 2*10) = 3 *21 = 63  
        # (3 coords * (original coords + 2*L freqs))
        #view_dir_PE_dims = dimension of positionally encoded viewing directions , l=4: 3 * (1 + 2*4) = 27
        super().__init__()
        self.pos_enc_dim = pos_enc_dim
        self.dir_enc_dim = dir_enc_dim

        # first 4 layers before skip
        self.layers_before_skip = nn.Sequential(
            nn.Linear(pos_enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # next 3 layers: process after skip connection, inject 3d world coords here
        self.layers_after_skip = nn.Sequential(
            nn.Linear(256 + pos_enc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256), # last layer before branching  
        )

        # density branch (only depends on 3d world coord inputs)
        self.density_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.ReLU() # density myst be >= 0
        )

        # RGB branch (depends on viewing dir and 3d world coords inpuyts)
        self.rgb_layer1 = nn.Linear(256, 256)

        # now concatenating with ray direction
        self.rgb_layers_after_dir = nn.Sequential(
            nn.Linear(256 + dir_enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128,3),
            nn.Sigmoid()
        )      

    def forward(self, pos_enc ,dir_enc):
        original_shape = pos_enc.shape

        # flatten to 2d process ing if needed
        if pos_enc.ndim > 2:
            pos_enc_flat = pos_enc.reshape(-1, pos_enc.shape[-1])
            dir_enc_flat = dir_enc.reshape(-1, dir_enc.shape[-1])
        else:
            pos_enc_flat = pos_enc
            dir_enc_flat = dir_enc

        # process first 4 layers
        x = self.layers_before_skip(pos_enc_flat)

        # skip connection
        x = torch.cat([x, pos_enc_flat], dim=-1)

        # next 3 layers after skip connection
        x = self.layers_after_skip(x)

        # density branch
        sigma = self.density_head(x)

        # rgb branch
        rgb_features = self.rgb_layer1(x)
        rgb_features = torch.cat([rgb_features, dir_enc_flat], dim=-1)
        rgb = self.rgb_layers_after_dir(rgb_features)

        if len(original_shape) > 2:
            rgb = rgb.reshape(*original_shape[:-1],3)
            sigma = sigma.reshape(*original_shape[:-1],1)
        return rgb, sigma

def prepare_nerf_inputs(points, rays_d, L_pos=10, L_dir=4):
    N_rays, N_samples = points.shape[:2]
    # encode 3d sample coords
    pos_enc = positional_encoding(points, L=L_pos)

    # assume rays_d have been normalized and is a torch array

    # broadcast ray direction to all samples along the ray
    rays_d_expanded = rays_d.unsqueeze(1).expand(N_rays, N_samples,3)
    # same viewing direction for all pts along ray
    dir_enc = positional_encoding(rays_d_expanded, L=L_dir)

    return pos_enc, dir_enc


