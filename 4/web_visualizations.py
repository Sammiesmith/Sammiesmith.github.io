import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# citation: this visualization is gpt generated

def plot_training_preds_grid(preds_folder="./my_training_preds", save_path="./output_imgs/my_training_preds_grid.png"):
    """
    Loads up to 8 rendered training prediction PNGs and plots them in a grid.
    Filenames should be something like: pred_iter_200.png, pred_iter_400.png, etc.
    """
    # list image paths
    files = [f for f in os.listdir(preds_folder) if f.endswith(".png")]
    if len(files) == 0:
        print(f"❌ No PNG files found in folder: {preds_folder}")
        return

    # sort by numeric iteration if filenames contain iteration numbers
    try:
        files = sorted(files, key=lambda x: int("".join([c for c in x if c.isdigit()])))
    except:
        files = sorted(files)

    # load first 8
    files = files[:10]
    imgs = [np.array(Image.open(os.path.join(preds_folder, f))) for f in files]

    # plot grid (2 rows x 4 columns)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    for ax, img, fname in zip(axs, imgs, files):
        ax.imshow(img)
        ax.set_title(fname)
        ax.axis("off")

    # if fewer than 8 images, hide empty axes
    for ax in axs[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"✅ Saved grid visualization to {save_path}")

if __name__ == "__main__":
    plot_training_preds_grid(
        preds_folder="./my_training_preds",
        save_path="my_training_preds_grid.png"
    )
