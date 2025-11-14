import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from code.harris import get_harris_corners
from code.anms import adaptive_non_maximal_suppression
from code.extract_feature_descriptors import extract


INPUT_IMG_PATH = "./data/C1.jpg" 

# load image and convert to grayscale
img = io.imread(INPUT_IMG_PATH)
if img.ndim == 3:
    img = color.rgb2gray(img)

# compute Harris corners only
# h, coords = get_harris_corners(img, min_distance=50, corner_threshold=0.01)


# compute Harris + ANMS corners
h, coords = get_harris_corners(img, min_distance=5, corner_threshold=0.001)
anms_coords, radii = adaptive_non_maximal_suppression(coords, h, c_robust=0.9, num_interest_pts=500)

# extract descriptors using 5 pixel spacing
descriptors, valid_coords = extract(img, anms_coords, spacing=5, descriptor_size=8, blur_sigma=1, bells=False)

print(f"Extracted {len(descriptors)} descriptors")

# visualize a few descriptor patches
fig, axes = plt.subplots(2, 5, figsize=(10,4))
for ax, d in zip(axes.ravel(), descriptors[:10]):
    ax.imshow(d.reshape(8,8), cmap="gray")
    ax.axis('off')
plt.suptitle("8x8 MOPS Feature Descriptors (with 5 pixel sampling and bias/gain normalization)")
plt.show()