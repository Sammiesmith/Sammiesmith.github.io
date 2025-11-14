
import numpy as np
import os
DATA_DIR = 'C:/Users/sammi/Sammiesmith.github.io/2/data'
output_dir = 'C:/Users/sammi/Sammiesmith.github.io/2/data/part_2_1'
os.makedirs(output_dir, exist_ok=True)
from scipy.signal import convolve2d
import cv2
##########################################
# Part 2.1: Image Sharpening with Unsharp Masking
##########################################

def format_img(filename):
    input_image_path = os.path.join(DATA_DIR, filename)  
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(float) / 255.0
    return img


def details(img, filename):

    # smooth img w gaussian filter
    kernel_size = 9
    sigma = 2
    g1d = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = g1d @ g1d.T
    img_gaussian = convolve2d(img, gaussian_kernel, mode='same', boundary='symm')
    cv2.imwrite(os.path.join(output_dir, 'gaussian_' + filename), np.clip(np.abs(img_gaussian)*255, 0, 255).astype(np.uint8))

    # now get high frequencies by subtracting smoothed img from original img
    img_highfreq = img - img_gaussian

    # save high freq img

    #cv2.imwrite(os.path.join(output_dir, 'highfreq_' + filename), np.clip(img_highfreq*255, 0, 255).astype(np.uint8)) # issue: highfreq all black for low contrast imgs
    cv2.imwrite(os.path.join(output_dir, 'highfreq_' + filename), cv2.normalize(np.abs(img_highfreq), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) # normalize to enhance contrast

    return img_highfreq


def sharpen(img, img_highfreq, alpha=1.5):
    # sharpened img = og img + alpha * high frequencies
    sharpened_img = img + alpha * img_highfreq
    return sharpened_img


if __name__ == "__main__":

    filenames = ['taj.jpg', 'emilee.jpg']
    for name in filenames: 
        img = format_img(name)
        # get high frequency  image details
        img_highfreq = details(img, name)
        print(f"Saved high frequency image to {output_dir}")
        # get sharpened image
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            sharpened_img = sharpen(img, img_highfreq, alpha)
            cv2.imwrite(os.path.join(output_dir, f'sharpened_alpha{alpha}_' + name), np.clip(np.abs(sharpened_img)*255, 0, 255).astype(np.uint8))
            print(f"Saved sharpened image with alpha={alpha} to {output_dir}")

