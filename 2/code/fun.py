import numpy as np
import os
input_image_path = os.path.join('C:/Users/sammi/Sammiesmith.github.io/2/data/sammie.jpg')
output_dir = os.path.join('C:/Users/sammi/Sammiesmith.github.io/2/data/fun/sammie')
os.makedirs(output_dir, exist_ok=True)
from scipy.signal import convolve2d
import cv2

##############################################################
# PART 1.2 FINITE DIFFERENCE OPERATORS
#############################################################
print("Part 1.2: Finite difference operators...")
image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype(float) / 255.0

Dx = np.array([[1, 0, -1]])
Dy = np.array([[1], [0], [-1]])

# convolve img with finite difference ops
image_dx = convolve2d(image, Dx, mode='same', boundary='symm')
image_dy = convolve2d(image, Dy, mode='same', boundary='symm')

# compute gradient magnitudes
gradient_magnitude = np.sqrt(image_dx**2 + image_dy**2)

# make binary edge map
threshold = 0.01
edge_image = gradient_magnitude > threshold

# save results
cv2.imwrite(os.path.join(output_dir, 'dx.jpg'), np.clip(np.abs(image_dx)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'dy.jpg'), np.clip(np.abs(image_dy)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'gradient_magnitude.jpg'), np.clip(np.abs(gradient_magnitude)*255, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(output_dir, 'edge.jpg'), np.clip(np.abs(edge_image)*255, 0, 255).astype(np.uint8))

print("saved to data/part_1_2")
