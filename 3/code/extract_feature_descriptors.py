import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def extract(image, coords, spacing=5, descriptor_size=8, blur_sigma=1.0, bells=False):
    """
    goal: extract 8x8 bias/gain normalized feature descriptor. Use 5-pixel sampling spacing,
    
    descriptor grid size 8x8

    returns descriptors, valid_coords
    descriptors = array of flattened normalized descriptors
    valid_coords = array of coords that produced valid descriptors
    """

    # blur image to act like a higher pyramid level, to reduce aliasing
    image = gaussian_filter(image, blur_sigma)

    half_patch = (descriptor_size // 2) * spacing # 8 // 2 = 4 * 5 = 20 = half of 40x40 pixel patch
    descriptors =[]
    valid_coords =[]

    for y, x in zip(coords[0], coords[1]):
        y = float(y)
        x = float(x)

        # skip if the 40x40 window goes over the edge of the image
        if (y - half_patch < 0 or y + half_patch >= image.shape[0] or x - half_patch < 0 or x + half_patch >= image.shape[1]):
            continue

        if bells:
            # bilinear interpolation to sample floating point (x,y) locations to fix rounding issue
            # build the sampling grid (8x8 with 5 pixel spacing)
            sample_y = np.linspace(y - half_patch + spacing/2, y + half_patch - spacing/2, descriptor_size)
            sample_x = np.linspace(x - half_patch + spacing/2, x + half_patch - spacing/2, descriptor_size)

            y_by_y, x_by_x = np.meshgrid(sample_y, sample_x, indexing='ij')

            # then can do bilinear interpolation
            patch = map_coordinates(image, [y_by_y.ravel(), x_by_x.ravel()], order=1, mode='reflect').reshape(descriptor_size, descriptor_size)
        else:
            x, y = int(round(y)), int(round(x))
            if (y - half_patch < 0 or y + half_patch >= image.shape[0] or x - half_patch < 0 or x + half_patch >= image.shape[1]):
                continue
            patch = image[y - half_patch : y + half_patch : spacing, x - half_patch : x + half_patch : spacing ]
            if patch.shape != (descriptor_size, descriptor_size):
                continue
        # bias/gain normalization
        mean, std = np.mean(patch), np.std(patch)
        if std < 1e-5:
            continue
        patch_norm = (patch - mean) / std
        descriptors.append(patch_norm.flatten())
        valid_coords.append([y, x])


    return np.array(descriptors), np.array(valid_coords).T



