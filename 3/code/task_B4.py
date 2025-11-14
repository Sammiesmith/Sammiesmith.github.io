import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from code.harris import get_harris_corners
from code.anms import adaptive_non_maximal_suppression
from code.extract_feature_descriptors import extract
from code.feature_matching import match_features
from code.ransac import ransac
from code.visualize_matches import visualize_matches
from code.make_mosaic import stitch_and_blend, get_mosaic_bounds

# Task B4: RANSAC for robust homography -------------------------

# 0) load images --------------------------------------------------
print("Loading Images")
# load img pair
# img1 = color.rgb2gray(io.imread("./data/A1.jpg"))
# img2 = color.rgb2gray(io.imread("./data/A2.jpg"))
# img1_rgb = io.imread("./data/A1.jpg")
# img2_rgb = io.imread("./data/A2.jpg")


# img1 = color.rgb2gray(io.imread("./data/B1.jpg"))
# img2 = color.rgb2gray(io.imread("./data/B2.jpg"))

# img1_rgb = io.imread("./data/B1.jpg")
# img2_rgb = io.imread("./data/B2.jpg")

img1 = color.rgb2gray(io.imread("./data/C1.jpg"))
img2 = color.rgb2gray(io.imread("./data/C2.jpg"))
img1_rgb = io.imread("./data/C1.jpg")
img2_rgb = io.imread("./data/C2.jpg")


# 1) Get correspondences via Harris Corners + ANMS + Feature Matching --------------------------------
print("Getting Harris Corners + ANMS")
h1, c1 = get_harris_corners(img1, min_distance=5, corner_threshold=0.01)
h2, c2 = get_harris_corners(img2, min_distance=5, corner_threshold=0.01)
c1, _ = adaptive_non_maximal_suppression(c1, h1, num_interest_pts=2000)
c2, _ = adaptive_non_maximal_suppression(c2, h2, num_interest_pts=2000)

# get 8x8 feature descriptors
descriptors1, valid_coords1 = extract(img1, c1, bells=True)
descriptors2, valid_coords2 = extract(img2, c2, bells=True)

print("Descriptors1: ", descriptors1.shape)
print("Descriptors2: ", descriptors2.shape)

print("Getting Features")
# match w lowe ratio test
matches, scores = match_features(descriptors1, descriptors2, valid_coords1, valid_coords2, ratio_threshold=0.6)

# 2) Do RANSAC to find best transformation and inliers -----------------------------------
print("Performing RANSAC")
points1 = valid_coords1[:, matches[:, 0]].T # (N,2)
points2 = valid_coords2[:, matches[:, 1]].T # (N, 2)
best_H, best_inliers = ransac(points1, points2, num_iter=2000, threshold=3.0)

# ----------------------------------------------------------
# DEBUG: visualize RANSAC inliers vs outliers
# ----------------------------------------------------------
print(f"RANSAC kept {best_inliers.sum()} / {len(best_inliers)} matches ({best_inliers.sum()/len(best_inliers):.2%})")

inlier_pts1 = points1[best_inliers]
inlier_pts2 = points2[best_inliers]
outlier_pts1 = points1[~best_inliers]
outlier_pts2 = points2[~best_inliers]

# Combine images side-by-side
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
combined = np.zeros((max(h1, h2), w1 + w2))
combined[:h1, :w1] = img1
combined[:h2, w1:w1 + w2] = img2

plt.figure(figsize=(12, 6))
plt.imshow(combined, cmap="gray")
plt.axis("off")

# Plot inliers (green)
for p1, p2 in zip(inlier_pts1, inlier_pts2):
    plt.plot([p1[1], p2[1] + w1], [p1[0], p2[0]], 'g-', linewidth=0.5)
# Plot outliers (red, thinner)
for p1, p2 in zip(outlier_pts1, outlier_pts2):
    plt.plot([p1[1], p2[1] + w1], [p1[0], p2[0]], 'r-', linewidth=0.3, alpha=0.4)

plt.title("RANSAC: Inliers (green) vs Outliers (red)")
plt.show()


# 3) Automaically Stitch mosaic using RANSAC results --------------------------------------------
print("Stitching Mosaic")

# TEST: warp img1 onto img2's size using best_H (no blending yet)
from code.warp_image import warpImageBilinear
test = warpImageBilinear(img1_rgb, best_H, img2.shape[1], img2.shape[0])
plt.imshow((test * 255).astype(np.uint8), cmap='gray')
plt.title("img1 warped onto img2 frame (H: img1->img2)")
plt.show()

mosaic = stitch_and_blend(img1_rgb, img2_rgb, best_H)

print("Visualizing mosaic")
# visualize mosaic
plt.figure(figsize=(10, 8))
plt.imshow(mosaic)
plt.title(f"Automatically Stitched Final Mosaic")
plt.axis("off")
plt.show()

# plot correspondences ---
min_x, max_x, min_y, max_y = get_mosaic_bounds(img1, img2, best_H)
translate = np.array([[1, 0, -min_x],
                        [0, 1, -min_y],
                        [0, 0, 1]], dtype=np.float64)
H2 = translate @ best_H
img1_pts = points1[best_inliers]
img2_pts = points2[best_inliers]

# transform correspondence points
img2_pts_h = np.hstack([img2_pts, np.ones((len(img2_pts), 1))])
warped_pts = (H2 @ img2_pts_h.T)
warped_pts /= warped_pts[2]
warped_pts = warped_pts[:2].T
img1_pts_mosaic = img1_pts + np.array([-min_x, -min_y])

plt.figure(figsize=(10, 8))
plt.imshow(mosaic)
plt.scatter(img1_pts_mosaic[:, 0], img1_pts_mosaic[:, 1], color='red', s=30, label='img1_pts')
plt.scatter(warped_pts[:, 0], warped_pts[:, 1], color='blue', s=30, label='img2 warped')
plt.legend()
plt.title("Correspondence alignment on mosaic canvas")
plt.show()