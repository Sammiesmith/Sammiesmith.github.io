import matplotlib.pyplot as plt
import numpy as np

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