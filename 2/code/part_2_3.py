import numpy as np
import matplotlib.pyplot as plt
import os
DATA_DIR = 'C:/Users/sammi/Sammiesmith.github.io/2/data'
output_dir = 'C:/Users/sammi/Sammiesmith.github.io/2/data/part_2_3'
os.makedirs(output_dir, exist_ok=True)
from scipy.signal import convolve2d
import cv2
##########################################
# Part 2.3: gaussian and laplacian stacks - the oraple journey
##########################################

# implement a gaussian and a laplacian stack 
# imgs never downsampled
# gaussian stack: apply gaussian filter at each level, but do not subsample
# laplacian stack: subtract each level of the gaussian stack from the previous level of the gaussian stack

def get_half_img(img, show_left_half=True):
    return ...

# return a gaussian filter
def get_gaussian_filter(sigma):
    kernel_size = int(2*np.ceil(3*sigma) + 1)
    g1d = cv2.getGaussianKernel(kernel_size, sigma)
    return g1d @ g1d.T

# blur image using gaussian
def apply_filter(img, filter):
    filtered_img = cv2.filter2D(img, -1, filter)
    return filtered_img

def build_gaussain_stack(img, sigma_list):
    stack = [img]
    for sigma in sigma_list:
        filter = get_gaussian_filter(sigma)
        next_img = apply_filter(img, filter)
        stack += [next_img]
    return stack

def build_laplacian_stack(gaussian_stack):
    # subtrackt next blur from prior blur to get laplacian stack
    laplacian_stack = []
    for i in range(0, len(gaussian_stack)-1):
        laplacian_stack += [gaussian_stack[i] - gaussian_stack[i+1]]
    return laplacian_stack

def normalize01(x):
    # normalize an array to range [0,1] for visualization
    x_min, x_max = np.min(x), np.max(x)
    if x_max - x_min < 1e-5:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def load_rgb_float(filename):
    # load img and return as rgb float [0,1]
    img = cv2.imread(os.path.join(DATA_DIR, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def build_mask(img, sigma, show_left_half=True):
    # return a mask w 1 on left , 0 on right, blurred w gaussian sigma
    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.float32)
    if show_left_half:
        mask[:, :w//2] = 1.0
    else:
        mask[:, w//2:] = 1.0
    filter = get_gaussian_filter(sigma)
    mask = cv2.filter2D(mask, -1, filter)
    # expand to 3 channels
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return mask

def build_soft_half_mask(img, transition_width_px, left_half=True):
    h, w = img.shape[:2]
    center_x = w // 2
    half_width = max(1, transition_width_px // 2)

    # horizontal coords
    x = np.arange(w, dtype=np.float32)

    # get center window [center_x - half_width, center_x + half_width]
    window = np.clip((x - (center_x - half_width)) / (2.0 * half_width), 0.0, 1.0)
    
    # smooth ramp using a small cosine to go from 0 to 1 soothly over transition window
    smooth_ramp = 0.5 - 0.5 * np.cos(np.pi * window)
    if left_half:
        smooth_ramp = 1.0 - smooth_ramp
    
    # expand to 2d and 3 channels
    blend_2d = np.tile(smooth_ramp[None, :], (h, 1)).astype(np.float32)
    blend_3d = np.repeat(blend_2d[:, :, np.newaxis], 3, axis=2)
    return blend_3d
    


def build_mask_stack(img, sigma_list, transition_width_px, left_half=True, extra_blur_sigma=0.0):
    base_mask = build_soft_half_mask(img, transition_width_px, left_half)
    mask_stack = []
    for sigma in sigma_list:
        # blur the base mask with extra_blur_sigma
        sigma_total = np.sqrt(sigma**2 + extra_blur_sigma**2) if extra_blur_sigma > 0 else sigma
        if sigma_total > 0.0:
            filter = get_gaussian_filter(sigma_total)
            # blur 1 channel then copy to 3 channels
            blurred_chan1 = cv2.filter2D(base_mask[:, :, 0], -1, filter)
            mask = np.repeat(blurred_chan1[:, :, np.newaxis], 3, axis=2)
            # normalize to [0,1]
            mask = normalize01(mask)
            mask_stack.append(mask)
        else:
            mask_stack.append(base_mask)
    return mask_stack

def get_stacks(img, sigma_list):
    g_stack = build_gaussain_stack(img, sigma_list)
    l_stack = build_laplacian_stack(g_stack)
    return g_stack, l_stack

def build_everything(img_a, img_b, sigma_list):
    g_stack_a, l_stack_a = get_stacks(img_a, sigma_list)
    g_stack_b, l_stack_b = get_stacks(img_b, sigma_list)
    mask_stack = build_mask_stack(img_a, sigma_list, transition_width_px=150, left_half=True, extra_blur_sigma=0.0)
    l_stack_sum = np.zeros(len(l_stack_a), dtype=object)
    g_stack_sum = np.zeros(len(g_stack_a), dtype=object)


    # per level contributions
    for level in [0,2,4]:
        # level mask
        mask = mask_stack[level]
        invs_mask = 1.0 - mask
        # level laplacian
        l_a = mask * l_stack_a[level]
        l_b = invs_mask * l_stack_b[level]
        l_sum = l_a + l_b
        # level gaussian
        g_a = mask * g_stack_a[level]
        g_b = invs_mask * g_stack_b[level]
        g_sum = g_a + g_b
        # alter imgs for display using plt
        l_stack_a[level] = normalize01(l_a)
        l_stack_b[level] = normalize01(l_b)
        l_stack_sum[level] = normalize01(l_sum)
        g_stack_a[level] = np.clip(g_a, 0, 1)
        g_stack_b[level] = np.clip(g_b, 0, 1)
        g_stack_sum[level] =np.clip(g_sum, 0, 1)

    return g_stack_a, g_stack_b, g_stack_sum, l_stack_a, l_stack_b, l_stack_sum

def display_everything(img_a, img_b, g_stack_a, g_stack_b, g_stack_sum, l_stack_a, l_stack_b, l_stack_sum):
    # citing source: used llm to generate this plotting code
    ####################################################################################
    # pick indices 0,2,4 safely
    possible_idxs = [0, 2, 4]
    g_idxs = [i for i in possible_idxs if i < len(g_stack_sum)]
    l_idxs = [i for i in possible_idxs if i < len(l_stack_sum)]
    print(g_idxs, l_idxs)

    n_g = len(g_idxs)
    n_l = len(l_idxs)

    # rows: 1 original + n_g Gaussian + n_l Laplacian
    n_rows, n_cols = 1 + n_g + n_l, 3
    fig_h = max(6, 2 + 2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, fig_h))

    # normalize axes for indexing
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    def _imshow(ax, img, *, cmap_default="gray"):
        """Show 2D as grayscale, 3D as RGB; clip to [0,1] for safety."""
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap_default)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")

    # Row 0: originals
    _imshow(axes[0, 0], img_a)
    axes[0, 0].set_title("Image A")
    _imshow(axes[0, 1], img_b)
    axes[0, 1].set_title("Image B")
    axes[0, 2].axis("off")

    # Gaussian rows
    for r_i, idx in enumerate(g_idxs):
        r = 1 + r_i
        _imshow(axes[r, 0], g_stack_a[idx])
        axes[r, 0].set_title(f"Gaussian A — level {idx}")
        _imshow(axes[r, 1], g_stack_b[idx])
        axes[r, 1].set_title(f"Gaussian B — level {idx}")
        _imshow(axes[r, 2], g_stack_sum[idx])
        axes[r, 2].set_title(f"Gaussian Sum — level {idx}")

    # Laplacian rows
    lap_base = 1 + n_g
    for r_i, idx in enumerate(l_idxs):
        r = lap_base + r_i
        _imshow(axes[r, 0], l_stack_a[idx])
        axes[r, 0].set_title(f"Laplacian A — level {idx}")
        _imshow(axes[r, 1], l_stack_b[idx])
        axes[r, 1].set_title(f"Laplacian B — level {idx}")
        _imshow(axes[r, 2], l_stack_sum[idx])
        axes[r, 2].set_title(f"Laplacian Sum — level {idx}")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    img_a = load_rgb_float('apple.jpeg')
    img_b = load_rgb_float('orange.jpeg')
    display_everything(img_a, img_b, *build_everything(img_a, img_b, sigma_list=[2,4,8,16,32, 64]))
    
    





