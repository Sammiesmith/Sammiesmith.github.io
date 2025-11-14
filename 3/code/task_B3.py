import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from code.harris import get_harris_corners
from code.anms import adaptive_non_maximal_suppression
from code.extract_feature_descriptors import extract
from code.feature_matching import match_features


def visualize_matches(image1, image2, coords1, coords2, matches, max_display=50):
    # side by side visualize matched features for two images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    combined = np.zeros((max(h1, h2), w1 + w2), dtype=np.float32)
    combined[:h1, :w1] = image1
    combined[:h2, w1:w2 + w2] = image2

    plt.figure(figsize=(12, 6))
    plt.imshow(combined, cmap="gray")
    plt.axis('off')

    shown = matches[:max_display]
    for (i1, i2) in shown:
        y1, x1 = coords1[:, i1]
        y2, x2 = coords2[:, i2]

        plt.plot([x1, x2 + w1], [y1, y2], 'r-', linewidth=0.5)
        plt.plot(x1, y1, 'go', markersize=3)
        plt.plot(x2 + w1, y2, 'bo', markersize=3)

    plt.title(f"{len(matches)} feature matches (displaying {min(max_display, len(shown))})")
    plt.show()


# load img pair
# img1 = color.rgb2gray(io.imread("./data/A1.jpg"))
# img2 = color.rgb2gray(io.imread("./data/A2.jpg"))

img1 = color.rgb2gray(io.imread("./data/B1.jpg"))
img2 = color.rgb2gray(io.imread("./data/B2.jpg"))

# img1 = color.rgb2gray(io.imread("./data/C1.jpg"))
# img2 = color.rgb2gray(io.imread("./data/C2.jpg"))

# Harris + ANMS
h1, c1 = get_harris_corners(img1, min_distance=5, corner_threshold=0.01)
h2, c2 = get_harris_corners(img2, min_distance=5, corner_threshold=0.01)
c1, _ = adaptive_non_maximal_suppression(c1, h1, num_interest_pts=2000)
c2, _ = adaptive_non_maximal_suppression(c2, h2, num_interest_pts=2000)

# get 8x8 feature descriptors
descriptors1, valid_coords1 = extract(img1, c1, bells=True)
descriptors2, valid_coords2 = extract(img2, c2, bells=True)

print("Descriptors1: ", descriptors1.shape)
print("Descriptors2: ", descriptors2.shape)

# match w lowe ratio test
matches, scores = match_features(descriptors1, descriptors2, valid_coords1, valid_coords2, ratio_threshold=0.6)

# visualize
visualize_matches(img1, img2, valid_coords1, valid_coords2, matches)