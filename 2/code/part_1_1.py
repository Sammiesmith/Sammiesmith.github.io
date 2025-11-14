import numpy as np
import os
input_image_path = os.path.join('..', '2', 'data', 'selfie.jpg')
output_dir = os.path.join('..', '2', 'data', 'part_1_1')
os.makedirs(output_dir, exist_ok=True)

##############################################################
# PART 1.1 CONVOLUTIONS FROM SCRATCH
#############################################################

# Convolutions using 4 for loops (with padding & 0 fill values) ---------------------------
def convolution_four_loops(img, kernel):
    kernel = np.flipud(np.fliplr(kernel)) # invert kernel up-down and left-right for convolution
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # pad w 0s
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output_img = np.zeros_like(img, dtype=float)

    for i in range(h):
        for j in range(w):
            pixel_update = 0.0
            for k in range(kh):
                for l in range(kw):
                    pixel_update += img_padded[i + k, j + l] * kernel[k, l]

            output_img[i, j] = pixel_update
    return output_img


# Convolutions using 2 for loops (with padding & 0 fill values) -----------------------
def convolution_two_loops(img, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # pad w 0s
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output_img = np.zeros_like(img, dtype=float)

    for i in range(h):
        for j in range(w):
            overlapping_patch = img_padded[i : i + kh, j : j + kw]
            output_img[i, j] = np.sum(overlapping_patch * kernel)
    return output_img

# Compare with scipy convolve function on a dummy image ----------------------------------
from scipy.signal import convolve2d


# use a dummy img array for comparison
image = np.random.randn(64, 64)
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) # smoothing filter

out_4 = convolution_four_loops(image, kernel)
out_2 = convolution_two_loops(image, kernel)
out_scipy = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

print("Part 1.1: Comparing convolution implementations...")
print("Max difference 4-loop vs SciPy convolution:", np.abs(out_4 - out_scipy).max())
print("Max difference 2-loop vs SciPy convolution:", np.abs(out_2 - out_scipy).max())

# Applying filters to selfie --------------------------------------------------------------
print("Part 1.1: Applying filters to Selfie...")

import cv2

image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype(float) / 255.0

box_filter = np.ones((9, 9)) *  (1/81)

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

image_box = convolution_two_loops(image, box_filter)
image_dx = convolution_two_loops(image, Dx)
image_dy = convolution_two_loops(image, Dy)

cv2.imwrite(os.path.join(output_dir, 'selfie_box.jpg'), (image_box * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'selfie_dx.jpg'), (np.abs(image_dx) * 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'selfie_dy.jpg'), (np.abs(image_dy) * 255).astype(np.uint8))

print("Saved Filtered images to, ", output_dir)


