import numpy as np
import cv2
import os
import sys
import glob

DATA_DIR = os.path.join('..', 'data')
OUTPUT_DIR = os.path.join('..', 'outputs_new')

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

def stretch(img, p1=1, p2=99): # kinda like histogram equalization i think
    # stretch img hist to [0, 255] based on percentiles p1, p2
    assert 0 <= p1 < p2 <= 100
    lower = np.percentile(img, p1)
    upper = np.percentile(img, p2)
    stretched = ((img - lower) / (upper - lower + 1e-10))
    stretched = np.clip(stretched, 0, 1)
    return (stretched * 255).astype(np.uint8)

def process_image(path, method='ncc', stretch_bool=False):
    align_method = method  # 'l2' or 'ncc'
    max_shift = 15
    border = 10  # crop border for alignment metric computation

    file_name = os.path.basename(path)
    print(f"processing {file_name}...")

    # load grayscale imgs 
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  skipping {file_name}, not found or cannot be opened.")
    

    blue, green, red = split_channels(image)

    # normalize channels to [0, 255]... (histogram equalization simple version)
    if stretch_bool:
        blue = stretch(blue, 10, 90)
        green = stretch(green, 1, 99)
        red = stretch(red, 10, 90)

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
    base_file_name = os.path.splitext(file_name)[0]
    # if stretch_bool:
    #     base_file_name += "_stretch"
    # base_file_name += f"_{align_method}"

    output_path = os.path.join(OUTPUT_DIR, base_file_name + '.jpg')
    written = cv2.imwrite(output_path, final_image)

    if not written:
        absolute_path = os.path.abspath(output_path)
        raise RuntimeError(f"cs2.imwrite failed... does this path exist? {absolute_path}")
    print(f"saved aligned image to {output_path}")

def main():
    ensure_dir(os.path.join(OUTPUT_DIR))

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else 'ncc'
        stretch =  sys.argv[3].lower() in ('true', '1', 'yes') if len(sys.argv) > 3 else False
        process_image(os.path.join(DATA_DIR, input_file), method=method, stretch_bool=stretch)
        sys.exit(0)

    else:
        method = 'ncc'  # default
        stretch = True # default
        files = []
        files += glob.glob(os.path.join(DATA_DIR, '*_naive.jpg')) 
        if not files:
            print("no input files found in", DATA_DIR)
            return
        for p in sorted(files):
            process_image(p, method=method, stretch_bool=stretch)

## main function
if __name__ == "__main__":
    main()