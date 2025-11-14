import os
import json
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Source: ChatGPT5 generated code to help plot pre-calculated correspondences on imgs for display purposes.

def plot_correspondences(image1_path, image2_path, correspondence_json_path):
    """
    Visualize point correspondences between two images.
    
    Parameters
    ----------
    image1_path : str
        Path to the first image (e.g. './data/C1.jpg')
    image2_path : str
        Path to the second image (e.g. './data/C2.jpg')
    correspondence_json_path : str
        Path to JSON file of correspondences in format:
        {
          "im1_name": "C1",
          "im2_name": "C2",
          "im1Points": [[x1, y1], [x2, y2], ...],
          "im2Points": [[x1', y1'], [x2', y2'], ...]
        }
    """

    # -----------------------------
    # Load the images
    # -----------------------------
    im1 = cv2.imread(image1_path)
    im2 = cv2.imread(image2_path)
    if im1 is None or im2 is None:
        raise FileNotFoundError("One or both image paths are invalid. Check your ./data folder.")

    # Convert BGR (OpenCV default) → RGB (matplotlib uses RGB)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Load the correspondences JSON
    # -----------------------------
    with open(correspondence_json_path, 'r') as f:
        corr = json.load(f)

    pts1 = np.array(corr["im1Points"])
    pts2 = np.array(corr["im2Points"])

    if pts1.shape != pts2.shape:
        raise ValueError("im1Points and im2Points must have same number of correspondences")

    n_points = pts1.shape[0]
    print(f"Loaded {n_points} correspondences between {corr['im1_name']} and {corr['im2_name']}.")

    # -----------------------------
    # Create a side-by-side visualization
    # -----------------------------
    # Concatenate images horizontally
    combined = np.hstack((im1, im2))

    # Compute width offset (since im2 is placed to the right of im1)
    width1 = im1.shape[1]

    plt.figure(figsize=(16, 8))
    plt.imshow(combined)
    plt.axis('off')

    # -----------------------------
    # Plot each correspondence pair
    # -----------------------------
    for i in range(n_points):
        (x1, y1) = pts1[i]
        (x2, y2) = pts2[i]
        x2_shifted = x2 + width1  # shift im2 coords to match concatenated layout

        color = [random.random(), random.random(), random.random()]  # random RGB color

        # Draw circles at each correspondence point
        plt.scatter(x1, y1, s=40, c=[color], marker='o', edgecolors='black', linewidths=0.5)
        plt.scatter(x2_shifted, y2, s=40, c=[color], marker='x', linewidths=1.5)

        # Draw line connecting the two points
        plt.plot([x1, x2_shifted], [y1, y2], color=color, linestyle='-', linewidth=1.0, alpha=0.6)

    plt.title(f"Point Correspondences: {corr['im1_name']} ↔ {corr['im2_name']}")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    data_dir = "./data"
    img_pair_name = 'C'

    # Adjust these names based on your dataset
    json_path = os.path.join(data_dir, img_pair_name + "_correspondences.json")

    # Load image names directly from JSON (keeps flexible)
    with open(json_path, 'r') as f:
        corr = json.load(f)
    im1_path = os.path.join(data_dir, img_pair_name + "1.jpg")
    im2_path = os.path.join(data_dir,  img_pair_name + "2.jpg")

    plot_correspondences(im1_path, im2_path, json_path)
