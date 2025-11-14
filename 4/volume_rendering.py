import torch
import torch.nn as nn
import numpy as np

def volume_rendering(rgb, sigma, t_vals):
    """
    Args:
    rgb = rgb colors at sample pts (N_rays, N_samples, 3)
    sigma = densities at sample pts (N_rays, N_samples, 1)
    t_vals = distance values along rays (N_rays, N_samples)

    Returns:
    rendered_colors = final rgb colors for each ray (N_rays, 3)
    """

    N_rays, N_samples = rgb.shape[:2]

    # conpute delta = dists between samples
    deltas = t_vals[:, 1:] - t_vals[:, :-1] # (N_rays, N_samples)
    # append large value to last delta to represent ray going to infinity
    deltas = torch.cat([deltas, torch.full((N_rays, 1), 1e10, device=deltas.device)], dim=-1)

    # compute alphas = probability of ray ending at each sample
    # alpha_i = 1 - exp(-sigma_i * delta_i)
    sigma = sigma.squeeze(-1)
    alpha = 1.0 - torch.exp(-torch.relu(sigma) * deltas) # debug: add relu for numerical stability

    # compute transmittance, probability of reaching sample i
    # T_i = exp(-sum over j<i of sigma_j * delta_j)
    
    # first comput 1-alpha for all samples
    one_minus_alpha = 1.0 -alpha
    # prepend 1.0 at beginning bc ray should start w full transmittance
    one_minus_alpha_shifted = torch.cat([torch.ones((N_rays, 1), device=one_minus_alpha.device), one_minus_alpha[:, :-1]], dim=-1)

    # compute cumulative product to get transmittance
    T = torch.cumprod(one_minus_alpha_shifted, dim=-1)

    # compute weights for each sample: T_i * alpha_i
    weights = T * alpha

    # final color is a weighted sum
    weights = weights.unsqueeze(-1)
    rendered_colors = torch.sum(weights * rgb, dim=1)

    return rendered_colors



