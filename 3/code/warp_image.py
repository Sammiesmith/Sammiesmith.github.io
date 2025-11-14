import numpy as np
import cv2
import matplotlib.pyplot as plt
from code.recover_homographies import computeH
import json

# Compute the output canvas size
# by projecting 4 corners of input img to see where they land... this is then defines the bounding box of the warped img
def get_bounds(image, H):
    h = image.shape[0]
    w = image.shape[1] # DEBUG: imgs read as height, width, channel NOT width, height, channel

    corners = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]]).T #(3,4)

    warped_corners = H @ corners 
    warped_corners /= warped_corners[2, :] # normalize by dividing by last coordinate so that last coordinate stays 1

    newx = warped_corners[0, :]
    newy = warped_corners[1, :]

    # return minx, maxx, miny, maxy
    return int(np.floor(newx.min())), int(np.ceil(newx.max())), int(np.floor(newy.min())), int(np.ceil(newy.max()))


# Nearest neighbor warping # DOUBLE FOR LOOP TOO SLOW NEED TO VECTORIZE
# def warpImageNearestNeighbor(image, H):
    # inverse warping + NN interpolation
    min_x, max_x, min_y, max_y = get_bounds(image, H) # find bounding box for warped img
    out_w = max_x - min_x
    out_h = max_y - min_y

    H_inverse = np.linalg.inv(H) # get inverse homography

    warped_image = np.zeros((out_h, out_w, 3), dtype=np.uint8) # initialize warped image w 0s

    # loop thru each pixel in the destination
    for y_out in range(out_h):
        for x_out in range(out_w):
            # convert t0 img coords in original img
            x_prime = x_out + min_x
            y_prime = y_out + min_y
            # now actually get the corresponding coords in original img
            og_coords = H_inverse @ np.array([x_prime, y_prime, 1])
            og_coords /= og_coords[2] # normalize so that last coord stays 1

            x_og = og_coords[0]
            y_og = og_coords[1]

            # still inside og image bounds?
            if 0 <= y_og < image.shape[0] and 0 <= x_og < image.shape[1]:
                # round to the nearest pixel (top left corner style)
                x_nn = int(round(x_og))
                x_nn = min(max(x_nn, 0), image.shape[1]-1) # force to be inside img bounds if not
                y_nn = int(round(y_og))
                y_nn = min(max(y_nn, 0), image.shape[0]-1) # force to be inside img bounds if not

                warped_image[y_out, x_out] = image[y_nn, x_nn]
    return warped_image

def warpImageNearestNeighbor(image, H):
    # inverse warping + NN interpolation
    min_x, max_x, min_y, max_y = get_bounds(image, H) # find bounding box for warped img
    out_w = max_x - min_x
    out_h = max_y - min_y

    H_inverse = np.linalg.inv(H) # get inverse homography

    # generate desination grid o pixels
    x_out, y_out = np.meshgrid(np.arange(out_w), np.arange(out_h))
    x_prime = x_out + min_x
    y_prime = y_out + min_y

    # flatten all pixel coordinatees to homog (3,N)
    destination_pts = np.stack([x_prime.ravel(), y_prime.ravel(), np.ones_like(x_prime).ravel()], axis=0)

    # map thru invs homog in singe matmul
    og_pts = H_inverse @ destination_pts
    og_pts /= og_pts[2,:]
    x_og = og_pts[0,:].reshape(out_h, out_w)
    y_og = og_pts[1,:].reshape(out_h, out_w)

    # nn rouding
    x_nn = np.rint(x_og).astype(int)
    y_nn = np.rint(y_og).astype(int)

    # misk of in bounds og pixels
    ok = (((x_nn >= 0) & (x_nn < image.shape[1])) & ((y_nn >= 0) & (y_nn < image.shape[0])))

    warped_image = np.zeros((out_h, out_w, 3), dtype=np.uint8) # initialize warped image w 0s
    warped_image[ok] = image[y_nn[ok], x_nn[ok]]

    return warped_image

# double foorloop too slow need to vectorize


def warpImageBilinear(image, H, out_w=None, out_h=None):
    # warp an img using a homography H with bilinear interpolation
    # H maps source --> destination (img1 -> img2)
    print("input img shape", image.shape, "ndim", image.ndim)

    image = image.astype(np.float32)
    if image.ndim == 2: # grayscale
        image = image[..., None] # (H, W, 1)
    print("input img shape after normalizing channel", image.shape, "ndim", image.ndim)
    
    h_in, w_in, num_channels = image.shape


    H_inverse = np.linalg.inv(H) # compute inverse homography
    
    # if output size not given, use input size
    if out_w is None: out_w = w_in
    if out_h is None: out_h = h_in

    # generate desination grid o pixels
    y_out_rows, x_out_cols = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')

    # flatten all pixel coordinatees to homog (3,N)
    homog_pts = np.stack([x_out_cols.ravel(), y_out_rows.ravel(), np.ones_like(x_out_cols).ravel()], axis=0)

    # map thru invs homog in singe matmul
    og_pts = H_inverse @ homog_pts
    og_pts /= og_pts[2,:]
    x_og = og_pts[0,:].reshape(out_h, out_w)
    y_og = og_pts[1,:].reshape(out_h, out_w)

    # get 4 nearest pts
    x0 = np.floor(x_og).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_og).astype(int)
    y1 = y0 + 1

    # ensure still in bounds by clipping
    x0 = np.clip(x0, 0, w_in- 1)
    x1 = np.clip(x1, 0, w_in - 1)
    y0 = np.clip(y0, 0, h_in - 1)
    y1 = np.clip(y1, 0, h_in - 1)

    xw = x_og - x0  # x weight # distances for weighting later
    yw = y_og - y0 # y weight
    
    warped_image = np.zeros((out_h, out_w, num_channels), dtype=np.float32)
    for channel in range(num_channels):
        # get pixel vals for 4 neighbors
        a = image[y0, x0, channel]
        b = image[y0, x1, channel]
        c = image[y1, x0, channel]
        d = image[y1, x1, channel]

        # copy formula from discussion sction
        warped_image[..., channel] = (
            a * (1 - xw) * (1 - yw) +
            b * xw * (1 - yw) +
            c * (1 - xw) * yw +
            d * xw * yw
        )

    # mask inbounds pixels
    ok = (x_og >= 0) & (x_og < w_in - 1) & (y_og >= 0) & (y_og < h_in - 1)
    warped_image[~ok, :] = 0
    warped_image = np.clip(warped_image, 0, 255).astype(np.uint8)
    if num_channels == 1:
        warped_image = warped_image[..., 0]

    print("warped output img shape", warped_image.shape, "ndim", warped_image.ndim)
    return warped_image








if __name__ == "__main__":
    img_file_name = "C1.jpg"
    
    image = cv2.cvtColor(cv2.imread("./data/" + img_file_name), cv2.COLOR_BGR2RGB)

    with open("./data/" + img_file_name[0] + "_correspondences.json", "r") as f: # correspondence files must be in the format: [name]_correspondences.json
        correspondences = json.load(f)
        
    img1_pts = np.array(correspondences["im1Points"])
    img2_pts = np.array(correspondences["im2Points"])
    
    H = computeH(img1_pts, img2_pts)
    print("Interpolating on Nearest Neighbors...")
    warped_nn = warpImageNearestNeighbor(image, H)
 
    print("Bilinear Interpolation in progress...")
    warped_bi = warpImageBilinear(image, H)

     # Display comparison
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.imshow(warped_nn)
    plt.title("Nearest Neighbor Warp")
    plt.axis('off')
    

    plt.subplot(1,2,2)
    plt.imshow(warped_bi)
    plt.title("Bilinear Warp")
    plt.axis('off')

    plt.show()






