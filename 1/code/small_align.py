import numpy as np
import cv2
import os
import sys

DATA_DIR = os.path.join('..', 'data')
OUTPUT_DIR = os.path.join('..', 'outputs')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_channels(image):
    height = image.shape[0] // 3
    blue = image[0:height, :]
    green = image[height:2*height, :]
    red = image[2*height:3*height, :]
    return blue, green, red

def crop(img, border=10):
    h, w = img.shape
    return img[border: h - border, border: w - border]

def align_l2(base, target, max_shift=15, border=10):
    base = crop(base, border).astype(np.float32)
    best_offset = (0,0)
    best_score = float('inf')

    for x_shift in range(-max_shift, max_shift + 1):
        for y_shift in range(-max_shift, max_shift + 1):
            shifted_target = np.roll(target, shift=(y_shift, x_shift), axis=(0, 1))
            shifted_target = crop(shifted_target, border).astype(np.float32)

            score = np.sum((base - shifted_target) ** 2) #l2 norm

            if score < best_score:
                best_score = score
                best_offset = (y_shift, x_shift)

    return best_offset

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    # return scalar fraction, higher means more similar
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    return float(np.sum(a * b) / denominator)

def align_ncc(base, target, max_shift=15, border=10):
    base = crop(base, border).astype(np.float32)
    best_offset = (0,0)
    best_score = -1.0 

    for x_shift in range(-max_shift, max_shift + 1):
        for y_shift in range(-max_shift, max_shift + 1):
            shifted_target = np.roll(target, shift=(y_shift, x_shift), axis=(0, 1))
            shifted_target = crop(shifted_target, border).astype(np.float32)

            score = ncc(base, shifted_target) #ncc score

            if score > best_score:
                best_score = score
                best_offset = (y_shift, x_shift)

    return best_offset

def reconstruct(blue, green, red, offsets): # just shift r and g not b
    aligned_green = np.roll(green, shift=offsets[0], axis=(0, 1))
    aligned_red = np.roll(red, shift=offsets[1], axis=(0, 1))
    return np.dstack([blue, aligned_green, aligned_red])

def crop_final(img, border=0):
    h, w, _ = img.shape
    return img[border: h - border, border: w - border, :]

## main function
if __name__ == "__main__":
    ensure_dir(os.path.join(OUTPUT_DIR))

    from glob import glob
    input_files = glob(os.path.join(DATA_DIR, '*.jpg'))
    print(f"found {len(input_files)} input files")

    align_method = 'ncc' # 'l2' or 'ncc'
    max_shift = 30
    border = 50


    for in_path in input_files:
        file_name = os.path.basename(in_path)
        print(f"processing {file_name}...")

        # load grayscale imgs 
        image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  skipping {file_name}, not found or cannot be opened.")
            continue

    
    # file_name = sys.argv[1] if len(sys.argv) > 1 else 'tobolsk.jpg'
    # max_shift = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    # border = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    # align_method = sys.argv[4] if len(sys.argv) > 4 else 'ncc' # l2 or ncc

    # load img as grayscale
    # print("loading image...")
    # image_path = os.path.join(DATA_DIR, file_name)
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if image is None:
    #     raise FileNotFoundError(f"Image file '{image_path}' not found.")

        print("aligning image...")
        blue, green, red = split_channels(image) # split channels
        if align_method == 'l2':
            green_offset = align_l2(blue, green, max_shift, border) # align g and r to b
            red_offset = align_l2(blue, red, max_shift, border)
        elif align_method == 'ncc':
            green_offset = align_ncc(blue, green, max_shift, border) # align g and r to b
            red_offset = align_ncc(blue, red, max_shift, border)
        else:
            raise NotImplementedError(f"Alignment method '{align_method}' is not implemented. Only 'l2' and 'ncc' are available.")

        
        print(f"  G offset (dy, dx): {green_offset}")
        print(f"  R offset (dy, dx): {red_offset}")

        print("reconstructing image...")
        aligned_image = reconstruct(blue, green, red, (green_offset, red_offset))

        final_image = crop_final(aligned_image, border=7)

        # save output
        output_path = os.path.join(OUTPUT_DIR, file_name)
        written = cv2.imwrite(output_path, final_image)

        if not written:
            absolute_path = os.path.abspath(output_path)
            raise RuntimeError(f"cs2.imwrite failed... does this path exist? {absolute_path}")
        print(f"saved aligned image to {output_path}")