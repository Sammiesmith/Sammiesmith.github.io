import matplotlib.pyplot as plt
from .align_image_code import align_images
import numpy as np
import os
import cv2


# First load images
DATA_DIR = 'C:/Users/sammi/Sammiesmith.github.io/2/data'
#####################################################################
# img1_filename = 'DerekPicture.jpg'
# img2_filename = 'nutmeg.jpg'

# img1_filename = 'sammie1.jpg' # high freq
# img2_filename = 'adrian2.jpg' # low freq
# output_filename = 'adrian2_sammie.jpg'

img1_filename = 'caledonia_frown.jpg' # high freq
img2_filename = 'caledonia_smile.jpg' # low freq
output_filename = '_mona_caledonia.jpg'


# low sf
im1 = cv2.imread(os.path.join(DATA_DIR, img1_filename))/255.0

# high sf
im2 = cv2.imread(os.path.join(DATA_DIR, img2_filename))/255.0


# Next align images (this code is provided, but may be improved)
im2_aligned, im1_aligned = align_images(im2, im1)
im1_save = np.clip(im1_aligned, 0, 1)
im2_save = np.clip(im2_aligned, 0, 1)
aligned1_uint8 = (im1_save * 255).astype(np.uint8)
aligned2_uint8 = (im2_save * 255).astype(np.uint8)


cv2.imwrite(os.path.join(DATA_DIR, 'aligned' + img2_filename), aligned2_uint8)
cv2.imwrite(os.path.join(DATA_DIR, 'aligned' + img1_filename), aligned1_uint8)

# im1_aligned = cv2.imread(os.path.join(DATA_DIR, 'aligned' + img1_filename))/255.0
# im2_aligned = cv2.imread(os.path.join(DATA_DIR, 'aligned' + img2_filename))/255.0





####################################################################



# # low sf
# im1 = cv2.imread(os.path.join(DATA_DIR, 'DerekPicture.jpg'))/255.0

# # high sf
# im2 = cv2.imread(os.path.join(DATA_DIR, 'nutmeg.jpg'))/255.0

# # Next align images (this code is provided, but may be improved)
# im1_aligned, im2_aligned = align_images(im2, im1)
# cv2.imwrite(os.path.join(DATA_DIR, 'aligned' + 'DerekPicture.jpg'), im1_aligned)
# cv2.imwrite(os.path.join(DATA_DIR, 'aligned' + 'nutmeg.jpg'), im2_aligned)

# im2_aligned = cv2.imread(os.path.join('C:/Users/sammi/Sammiesmith.github.io/2/data', 'alignedDerekPicture.jpg'))/255.0
# im1_aligned = cv2.imread(os.path.join('C:/Users/sammi/Sammiesmith.github.io/2/data', 'alignednutmeg.jpg'))/255.0




im1_aligned = im1_aligned.astype(np.float32)
im2_aligned = im2_aligned.astype(np.float32)

# resize images to be same size
h, w, c = im1_aligned.shape
im2_aligned = cv2.resize(im2_aligned, (w, h)) 
im1_aligned = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2RGB)
im2_aligned = cv2.cvtColor(im2_aligned, cv2.COLOR_BGR2RGB)

#------------------------------------------------------------------------------------#
# low pass filter low freq img
def low_pass(img, sigma):
    kernel_size = int(2*np.ceil(3*sigma) + 1)
    g1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = g1d @ g1d.T
    img_low = cv2.filter2D(img, -1, gaussian_kernel)
    cv2.imwrite(os.path.join(DATA_DIR, f'gaussian_kernel_sigma{sigma}.jpg'), cv2.normalize(np.abs(img_low), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) 
    return img_low

# high pass filter high freq img
def high_pass(img, sigma):
    img_low = low_pass(img, sigma)
    img_high = img - img_low
    cv2.imwrite(os.path.join(DATA_DIR, f'gaussian_kernel_sigma{sigma}.jpg'), cv2.normalize(np.abs(img_high), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) 
    return img_high

# hybrid image function
def hybrid_image(im1, im2, sigma1, sigma2):
    # im1_high = high_pass(im1, sigma1)
    # im2_low = low_pass(im2, sigma2)
    #hybrid = im1_high + im2_low # doesnt work, dereks edges are too sharp
    # try adding some low pass derek
    im1_low = low_pass(im1, sigma1)
    im1_high = im1 - im1_low
    im2_low = low_pass(im2, sigma2)
    hybrid = 1.5 * im1_high + 0.5 *  im2_low + 0.3 * im1_low #1, 0.3 for derek, nugmeg ; 1.5, 0.5, 0.3 for sammie, adrian; 
    return hybrid

def fft_magnitude(img, name):
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitdue = np.log(np.abs(fshift) + 1e-8)
    # normalize to [0, 255]
    magnitdue = cv2.normalize(magnitdue, None, 0, 255, cv2.NORM_MINMAX)
    magnitdue = magnitdue.astype(np.uint8)
    cv2.imwrite(os.path.join(DATA_DIR, name), magnitdue)
    plt.imshow(magnitdue, cmap='gray')


## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

# sigma1 = 2.0 # high pass nutmeg
# sigma2 = 10.0 # low pass derek
sigma1 = 5.0 # high pass sammie
sigma2 = 10.0 # low pass adrian

hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

hybrid = np.clip(hybrid, 0, 1)

hybrid_uint8 = (hybrid * 255).astype(np.uint8)
aligned_hybrid_bgr = cv2.cvtColor(hybrid_uint8, cv2.COLOR_RGB2GRAY)

h,w = hybrid.shape[:2]
x_start = int(w *0.25) # left boundary
x_end = int(w * 0.75) # right boundary
y_start = int(h * 0.30) # top boundary
y_end = int(h * 0.85) # bottim boundary
hybrid = hybrid[y_start:y_end, x_start:x_end]
cv2.imwrite(os.path.join(DATA_DIR, 'hybrid' + output_filename), aligned_hybrid_bgr)
fft_magnitude(hybrid, 'hybrid_fft_magnitude' + output_filename)
fft_magnitude(im1_aligned, 'highfreq_fft_magnitude' + img1_filename)
fft_magnitude(im2_aligned, 'lowfreq_fft_magnitude' + img2_filename)


print("Saved aligned images to data directory.")




## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
# N = 5 # suggested number of pyramid levels (your choice)
# pyramids(hybrid, N)