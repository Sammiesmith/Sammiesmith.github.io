import numpy as np
import os
input_image_path = os.path.join('..', '2', 'data', 'cameraman.jpg')
output_dir = os.path.join('..', '2', 'data', 'part_1_3')
os.makedirs(output_dir, exist_ok=True)
from scipy.signal import convolve2d
import cv2

##############################################################
# PART 1.3 DERIVATIVE OF GAUSSIAN (DoG) FILTER
#############################################################
print("Part 1.3: Derivative of Gaussian (DoG) Filters...")
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype(float) / 255.0

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

# 1) SMOOTH IMAGE WITH GAUSSIAN
# Use built in gaussian filter on cameraman img 
kernel_size = 9
sigma = 2 
g1d = cv2.getGaussianKernel(kernel_size, sigma)
gaussian_kernel = g1d @ g1d.T

image_gaussian = convolve2d(image, gaussian_kernel, mode='same', boundary='symm')
image_dx_smooth =  convolve2d(image_gaussian, Dx, mode='same', boundary='symm')
image_dy_smooth =  convolve2d(image_gaussian, Dy, mode='same', boundary='symm')

# Then create gradient magnidue image
gradient_magnitude_smooth = np.sqrt(image_dx_smooth**2 + image_dy_smooth**2)

# And create binary edge map
threshold = 0.1
edge_smooth = gradient_magnitude_smooth > threshold

# save imgs


cv2.imwrite(os.path.join(output_dir, 'cameraman_dx_smooth.jpg'), np.clip(np.abs(image_dx_smooth)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'cameraman_dy_smooth.jpg'), np.clip(np.abs(image_dy_smooth)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'cameraman_gradient_magnitude_smooth.jpg'), np.clip(np.abs(gradient_magnitude_smooth)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'cameraman_edge_smooth.jpg'), np.clip(np.abs(edge_smooth)*255, 0, 255).astype(np.uint8))

# 2) DERIVATIVE OF GASSIAN FILTERS
# Use gaussian filter (by convolving gaussian with finite diff ops) on cameraman img 
DoG_x = convolve2d(gaussian_kernel, Dx, mode='same', boundary='symm')
DoG_y = convolve2d(gaussian_kernel, Dy, mode='same', boundary='symm')

cv2.imwrite(os.path.join(output_dir, 'DoGx_filter.jpg'), cv2.normalize(np.abs(DoG_x), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) #bug fixed: must normalize b/c DoG_x andy are too small to be cast directly to uint8 
cv2.imwrite(os.path.join(output_dir, 'DoGy_filter.jpg'), cv2.normalize(np.abs(DoG_y), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

image_dogx = convolve2d(image, DoG_x, mode='same', boundary='symm')
image_dogy = convolve2d(image, DoG_y, mode='same', boundary='symm')

# Then create gradient magnidue image
gradient_magnitude_dog = np.sqrt(image_dogx**2 + image_dogy**2)

# And create binary edge map
edge_dog = gradient_magnitude_dog > threshold

# save imgages
cv2.imwrite(os.path.join(output_dir, 'cameraman_dogx.jpg'), np.clip(np.abs(image_dogx)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'cameraman_dogy.jpg'), np.clip(np.abs(image_dogy)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'cameraman_gradient_magnitude_dog.jpg'), np.clip(np.abs(gradient_magnitude_dog)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'cameraman_edge_dog.jpg'), np.clip(np.abs(edge_dog)*255, 0, 255).astype(np.uint8))

print("saved to data/part_1_3")







