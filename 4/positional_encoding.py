import torch
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(coords, L):
    """
    This function applies sinusoidal positional encoding to coordinates.

    coords = a tensor w shape (batch size, dimension). For 2d: (batch size, 2) (uv coords). values normalized to [0,1]
    L = max freq level. Creating 2*L features per input dimension (L sine features + L cosine features)

    return encoded tensor of shape (batch size, dimensions + dimensions * (2 * L)). Ex: 2D w L=5 --> (batch size, 2 + (2* 2 * 5)) = (batchsize, 22) 
    --> include the +dimension bc keeping original feature in positional encoding as well.
    """
    # need to make sure that coords is a tensor so that downstream gradients work during training
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)

    encoded_features = []

    # for each freq level 0 to L-1
    for i in range(L):
        freq = 2**i * np.pi 
        encoded_features.append(torch.sin(freq * coords)) #(batch size, dim)
        encoded_features.append(torch.cos(freq * coords)) #(batch size, dim)

    # need to concatenate all features along the last dimension
    # ex: coords=(batchsize,2), L=5 --> 2*5=10 tensors with shape (batchsize,2)... concatenate gives (batchsize, 20)
    encoded_features = torch.cat(encoded_features, dim=-1) # -1 flattens

    # append the orginal feature
    encoded_features = torch.cat([coords, encoded_features], dim=-1)

    return encoded_features

def test_encoding():
    # make sure encodings are the right shape
    
    # test a single 2D coord [0.1, 0.1] with L = 5
    coords = torch.tensor([0.1, 0.1], dtype=torch.float32)
    encoded = positional_encoding(coords, 5)

    print(f"Expected encoding shape: (N, d + d*2*L) = (1, 2 +2*2*5) = (1, 22)")
    print(f"Positional Encoding shape: {encoded.shape} ")


if __name__ == "__main__":
    test_encoding()




    