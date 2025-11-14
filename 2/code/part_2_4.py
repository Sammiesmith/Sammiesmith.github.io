import numpy as np
import matplotlib.pyplot as plt
import os
DATA_DIR = 'C:/Users/sammi/Sammiesmith.github.io/2/data'
output_dir = 'C:/Users/sammi/Sammiesmith.github.io/2/data/part_2_4'
os.makedirs(output_dir, exist_ok=True)
from scipy.signal import convolve2d
import cv2
##########################################
# Part 2.4: multiresolution blending - the oraple journey
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

def build_rectuangular_mask(img, xrange, yrange, blur_sigma=60.0, invert=False):
    h, w = img.shape[:2]
    # rectangular mask:
    xmin, xmax = xrange
    ymin, ymax = yrange
    mask = np.ones((h,w), dtype=np.float32)
    mask[ymin:ymax, xmin:xmax] = 0.0
    #smooth edges w gaussian blur
    if blur_sigma > 0:
        filter = get_gaussian_filter(blur_sigma)
        mask = cv2.filter2D(mask, -1, filter)
    mask = normalize01(mask)
    if invert:
        mask = 1.0 - mask
    mask_rgb = np.repeat(mask[:,:,None], 3, axis=2)
    return mask_rgb    


def build_mask_stack(img, sigma_list, transition_width_px, left_half=True, extra_blur_sigma=0.0, mask_fn="rectangle", invert=False):
    if mask_fn == "half":
        base_mask = build_soft_half_mask(img, transition_width_px, left_half)
    elif mask_fn =="rectangle":
        base_mask = build_rectuangular_mask(img, (800, 1400), (450, 1340), blur_sigma=60.0, invert=invert)#build_rectuangular_mask(img, (950, 1600), (900, 1150), blur_sigma=100.0, invert=invert)
    else:
        raise Exception("unknown mask")
    mask_stack = [base_mask]
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

def build_everything(img_a, img_b, sigma_list, mask_fn):
    img_b = cv2.GaussianBlur(img_b, (0,0), sigmaX=3, sigmaY=3)
    g_stack_a, l_stack_a = get_stacks(img_a, sigma_list)
    g_stack_b, l_stack_b = get_stacks(img_b, sigma_list)
    mask_stack = build_mask_stack(img_a, sigma_list, transition_width_px=150, left_half=True, extra_blur_sigma=0.0, mask_fn=mask_fn, invert=False)

    # per level contributions
    for level in [0,2,4]:
        # level mask
        mask = mask_stack[level]
        invs_mask = 1.0 - mask
        # level laplacian
        l_a = mask * l_stack_a[level]
        l_b = invs_mask * l_stack_b[level]
        # level gaussian
        g_a = mask * g_stack_a[level]
        g_b = invs_mask * g_stack_b[level]
        # alter imgs for display using plt
        l_stack_a[level] = l_a
        l_stack_b[level] = l_b
        g_stack_a[level] = np.clip(g_a, 0, 1)
        g_stack_b[level] = np.clip(g_b, 0, 1)

    return g_stack_a, g_stack_b, l_stack_a, l_stack_b, mask_stack
def reconstruct(g_stack_a, g_stack_b, l_stack_a, l_stack_b, mask_stack):
    length = len(l_stack_a)

    # blend top gaussian level
    top = mask_stack[-1] * g_stack_a[-1] + (1.0 - mask_stack[-1]) * g_stack_b[-1]

    # recunstruct laplacian levels
    accumulate = top
    for i in range(length):
        accumulate += mask_stack[i] * l_stack_a[i] + (1.0 - mask_stack[i]) * l_stack_b[i]
    
    return np.clip(accumulate, 0.0, 1.0)

def display_laplacian_and_inputs(
    img_a: np.ndarray,
    img_b: np.ndarray,
    l_stack_a: list[np.ndarray],
    l_stack_b: list[np.ndarray],
    mask_stack: list[np.ndarray],
    levels_to_show: list[int] = [0, 2, 4]
):
    """
    Display masked input images and Laplacian stack levels.

    Inputs:
        img_a, img_b: original input images (H,W,3)
        l_stack_a, l_stack_b: Laplacian stacks for A and B (each a list of H,W,3 floats)
        mask_stack: list of masks for each level (H,W,3 floats in [0,1])
        levels_to_show: which Laplacian levels to visualize (indices)
    """
    # --- Make masked inputs for visualization ---
    # Using level 0 mask (least blurred)
    mask0 = mask_stack[0]
    inv_mask0 = 1.0 - mask0
    masked_a = mask0 * img_a
    masked_b = inv_mask0 * img_b

    # --- Determine rows and columns ---
    n_levels = len(levels_to_show)
    n_rows = 2 + n_levels  # row 0 = masked inputs, row 1+ = Laplacians
    n_cols = 3             # A, B, Mask or Combined

    fig_h = max(8, 2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, fig_h))

    def _imshow(ax, img, title=None, cmap='gray'):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(np.clip(img, 0, 1))
        if title:
            ax.set_title(title)
        ax.axis('off')

    # --- Row 0: masked inputs ---
    _imshow(axes[0, 0], masked_a, "Masked Image A")
    _imshow(axes[0, 1], masked_b, "Masked Image B")
    _imshow(axes[0, 2], mask0[...,0], "Mask Level 0", cmap='gray')

    # --- Subsequent rows: Laplacian levels ---
    for r_i, lvl in enumerate(levels_to_show):
        row_idx = r_i + 1
        if lvl >= len(l_stack_a):  # safety check
            break
        _imshow(axes[row_idx, 0], normalize01(l_stack_a[lvl]), f"Laplacian A lvl {lvl}")
        _imshow(axes[row_idx, 1], normalize01(l_stack_b[lvl]), f"Laplacian B lvl {lvl}")
        _imshow(axes[row_idx, 2], mask_stack[lvl][...,0], f"Mask lvl {lvl}", cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_a = load_rgb_float('mona_lisa.jpg')
    img_b = load_rgb_float('caledonia_mona_face.jpg')


    # g_stack_a, g_stack_b, l_stack_a, l_stack_b, mask_stack = build_everything(
    #     img_a, img_b,
    #     sigma_list=[2,4,8,16,32],
    #     mask_fn="rectangle"
    # )

    # display_laplacian_and_inputs(
    #     img_a, img_b,
    #     l_stack_a, l_stack_b, mask_stack,
    #     levels_to_show=[0,2,4]
    # )
    final = reconstruct(*build_everything(img_a, img_b, sigma_list=[2,4,8,16,32], mask_fn="rectangle"))
    plt.imsave(os.path.join(output_dir, 'final_blend_mona_caledonia.png'), (final*255).astype(np.uint8))
    plt.imshow(final)
    plt.axis('off')
    plt.title("Blended Image")
    plt.show()
    
    
    





