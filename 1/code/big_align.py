import os, sys, glob
import numpy as np
import cv2

# fixed params -------------------------------------------
METRIC = 'ncc' # 'l2' or 'ncc'
BORDER_CROP = 10 # pixels to crop from each border when aligning
COARSE_SEARCH = 48 # window at coarsest level
REFINE_SEARCH = 15 # window per finer level
MAX_LEVELS = 6 

DATA_DIR = os.path.join('..', 'data')
OUTPUT_DIR = os.path.join('..', 'outputs_new')
os.makedirs(OUTPUT_DIR, exist_ok=True)


#------------------------------- preprocessing ------------------------
def read_image_gray(path): # str -> np.ndarray
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0 
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img

def split_channels(image): 
    height = image.shape[0] // 3
    blue = image[0:height, :]
    green = image[height:2*height, :]
    red = image[2*height:3*height, :]
    return blue, green, red

# crop borders to stop edge plates from messing up the score
def crop_internal(img, border=10): 
    h, w = img.shape
    return img[border: h - border, border: w - border]

def sobel_edges(img): 
    # return image edges using sobel filter
    img = img.astype(np.float32)
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    return edges

def build_image_pyramid(img):
    # use gaussianblur and downsample to build pyramid of image from scratch
    blur = cv2.GaussianBlur(img, (5,5), 0) # apply 5x5 kernel
    return blur[::2, ::2] # downsample by factor of 2

#--------------------- scoring funcs ------------------
def ncc(a,b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    return float(np.sum(a * b) / denominator)

def l2(a,b):
    difference = a - b
    return float(np.sum(difference ** 2))

def get_overlapping_slices(h, w, y_shift, x_shift):
    # return slices for overlapping region of two images shifted by (y_shift, x_shift)
    if y_shift >= 0:
        y_start_base, y_end_base = y_shift, h
        y_start_target, y_end_target = 0, h - y_shift
    else:
        y_start_base, y_end_base = 0, h + y_shift
        y_start_target, y_end_target = -y_shift, h 

    if x_shift >= 0:
        x_start_base, x_end_base = x_shift, w
        x_start_target, x_end_target = 0, w - x_shift
    else:
        x_start_base, x_end_base = 0, w + x_shift
        x_start_target, x_end_target = -x_shift, w

    base_slice = (slice(y_start_base, y_end_base), slice(x_start_base, x_end_base))
    target_slice = (slice(y_start_target, y_end_target), slice(x_start_target, x_end_target))

    return base_slice, target_slice

def align_naive(base_img, target_img, max_shift=15, metric=METRIC, border_crop=BORDER_CROP, center_shift=None):
    # align target_img to base_img by exhausitively searching over patches in [-max_shift, max_shift]
    assert metric in ['l2', 'ncc']

    base_img = crop_internal(base_img, border_crop).astype(np.float32)
    target_img = crop_internal(target_img, border_crop).astype(np.float32)

    base_img = sobel_edges(base_img)
    target_img = sobel_edges(target_img)

    h, w = base_img.shape
    best_offset = (0,0) if center_shift is None else center_shift
    best_score = -1.0 if metric == 'ncc' else float('inf')

    if center_shift is None:
        y_shift_center, x_shift_center = 0, 0
    else:
        y_shift_center, x_shift_center = center_shift


    for x_shift in range(x_shift_center - max_shift, x_shift_center + max_shift + 1):
        for y_shift in range(y_shift_center - max_shift, y_shift_center + max_shift + 1):
            base_slice, target_slice  = get_overlapping_slices(h, w, y_shift, x_shift)

            # speedup... skip if not enough overlap
            if (base_slice[0].stop - base_slice[0].start < 20) or (base_slice[1].stop - base_slice[1].start < 20):
                continue

            base_patch = base_img[base_slice]
            target_patch = target_img[target_slice]

            if metric == 'ncc':
                score = ncc(base_patch, target_patch)
                if score > best_score:
                    best_score = score
                    best_offset = (y_shift, x_shift)
            else: # l2
                score = l2(base_patch, target_patch)
                if score < best_score:
                    best_score = score
                    best_offset = (y_shift, x_shift)

    return best_offset

def align_pyramid(base_img, target_img, metric=METRIC, border_crop=BORDER_CROP, 
                  coarse_window=COARSE_SEARCH, refine_window=REFINE_SEARCH, max_levels=MAX_LEVELS):
    # recursively align target_img to base_img using image pyramid (coarse to fine)
    # base case: if img is small enough, use naive alignment
    # recursive case: downsample images, align at lower res, then refine at current res
        # dwon sample base and target imgs by 2
        # recursively align at smaller scale 
        # scale shift up by two
        # refine shift at current scale in a local window around scaled shift

    # base case (img small enoigh)
    if max(base_img.shape[0], base_img.shape[1]) <= 400 or max_levels == 0:
        return align_naive(base_img, target_img, coarse_window, metric, border_crop)
    
    # recursive step on smaller imgs
    base_small = build_image_pyramid(base_img)
    target_small = build_image_pyramid(target_img)

    coarse_shift = align_pyramid(base_small, target_small, metric=metric, border_crop=border_crop,
                                 coarse_window=coarse_window, refine_window=refine_window, max_levels=max_levels - 1)

    # scale up shift by 2
    scaled_shift = (coarse_shift[0] * 2, coarse_shift[1] * 2)

    # refine at current scale w small local search
    refined_shift = align_naive(base_img, target_img, refine_window, metric, border_crop, center_shift=scaled_shift)

    return refined_shift

   
def reconstruct(blue, green, red, offsets): # just shift r and g not b
    aligned_green = np.roll(green, shift=offsets[0], axis=(0, 1))
    aligned_red = np.roll(red, shift=offsets[1], axis=(0, 1))
    return np.dstack([aligned_red, aligned_green, blue]) # rgb

def save_rgb_as_jpg(out_path, rgb):
    # save rgb image as jpg
    rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    written = cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not written:
        raise IOError(f"failed to write image to {out_path}")
    
# ------------------------------ main funcs ---------------------------------------
def process_image(path):
    img = read_image_gray(path)
    blue, green, red = split_channels(img)

    green_offset = align_pyramid(blue, green)
    red_offset = align_pyramid(blue, red)

    print(os.path.basename(path), "green offset:", green_offset, "red offset:", red_offset)

    bgr = reconstruct(blue, green, red, (green_offset, red_offset))

    out_name = os.path.splitext(os.path.basename(path))[0] + '_bells.jpg'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    save_rgb_as_jpg(out_path, bgr)
    print("  saved to", out_path)


def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        process_image(os.path.join(DATA_DIR, input_file))

    else:
        files = []
        files += glob.glob(os.path.join(DATA_DIR, '*.jpg'))
        files += glob.glob(os.path.join(DATA_DIR, '*.tif'))
        if not files:
            print("no input files found in", DATA_DIR)
            return
        for p in sorted(files):
            process_image(p)

if __name__ == "__main__":
    main()