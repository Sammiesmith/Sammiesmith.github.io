import matplotlib.pyplot as plt
from skimage import io, color
from code.harris import get_harris_corners, dist2
from code.anms import adaptive_non_maximal_suppression

INPUT_IMG_PATH = "./data/C1.jpg" 

# load image and convert to grayscale
img = io.imread(INPUT_IMG_PATH)
if img.ndim == 3:
    img = color.rgb2gray(img)

# compute Harris corners
h, coords = get_harris_corners(img, min_distance=50, corner_threshold=0.01)

# plot result for just harris
plt.figure(figsize=(8,8))
plt.imshow(img, cmap="gray")
plt.scatter(coords[1], coords[0], s=1, c='red', marker="o", label="Harris Corners")
plt.title("Harris Corners")
plt.axis('off')
plt.legend()
plt.show()

# compute Harris + ANMS corners
h, coords = get_harris_corners(img, min_distance=5, corner_threshold=0.001)
anms_coords, radii = adaptive_non_maximal_suppression(coords, h, c_robust=0.9, num_interest_pts=500)



# plot result for harris + anms
plt.figure(figsize=(8,8))
plt.imshow(img, cmap='gray')
plt.scatter(anms_coords[1], anms_coords[0], s=1, c='red', marker="o", label="ANMS Corners")
plt.title("Harris Corners using Adaptive Non-Maximal Supression")
plt.legend()
plt.axis('off')
plt.show()
