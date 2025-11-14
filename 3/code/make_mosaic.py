import numpy as np
import cv2
import matplotlib.pyplot as plt
from code.recover_homographies import computeH
from code.warp_image import warpImageBilinear
import json

# helper fn to make alpha mask
def get_alpha_mask(image):
    # 1 at center o img, fade to zero towards edges
    h = image.shape[0]
    w = image.shape[1]
    x, y = np.mgrid[0:h, 0:w] # coord grid for mask

    # now need to calc distance from each pixel to nearest edge for mask val
    dist_l = x
    dist_r = w - x - 1
    dist_u = y 
    dist_d = h - y - 1

    dist_to_closest_edge = np.minimum(np.minimum(dist_l, dist_r), np.minimum(dist_u, dist_d))
    alpha_mask = dist_to_closest_edge / dist_to_closest_edge.max() # normalize so largest value is 1 and smallest is 0
    alpha_mask_3_channel = np.dstack([alpha_mask]*3)
    return alpha_mask_3_channel

def get_mosaic_bounds(img1, img2, H):
    # project corners of both imgs and return bounding box for mosaic (global mosaic canvas, not warping one img onto the next)
    # warp img2 to img1 frame
    h1 = img1.shape[0]
    h2 = img2.shape[0]
    w1 = img1.shape[1]
    w2 = img2.shape[1]

    img1_corners = np.array([[0,0,1], [w1,0,1], [w1,h1,1], [0, h1, 1]], dtype=np.float64).T
    img2_corners = np.array([[0,0,1], [w2,0,1], [w2,h2,1], [0, h2, 1]], dtype=np.float64).T

    # need to warp the second img corners to align with first img coord system
    img2_corners_warped = H @ img2_corners
    img2_corners_warped /= img2_corners_warped[2] # normalize sp that last elem is still 1

    # combine da corners
    x = np.hstack([img1_corners[0], img2_corners_warped[0]])
    y = np.hstack([img1_corners[1], img2_corners_warped[1]])

    return int(np.floor(x.min())), int(np.ceil(x.max())), int(np.floor(y.min())), int(np.ceil(y.max()))

# # blend second img onto first img to make a mosaicc
# def stitch_and_blend(img1, img2, H):
#     min_x, max_x, min_y, max_y = get_mosaic_bounds(img1, img2, H)
#     out_w, out_h = max_x - min_x, max_y - min_y

#     # need to translate - minx, - min y so centered at 0 (see dic wksht)
#     translate = np.array([[1,0,-min_x], [0, 1, -min_y], [0, 0 , 1]], dtype=np.float64) 

#     H1 = translate
#     H2 = translate @ H

#     img1_warped = warpImageBilinear(img1, H1, out_w, out_h)
#     img2_warped = warpImageBilinear(img2, H2, out_w, out_h)

#     h_min = min(img1_warped.shape[0], img2_warped.shape[0])
#     w_min = min(img1_warped.shape[1], img2_warped.shape[1])
#     img1_warped = img1_warped[:h_min, :w_min]
#     img2_warped = img2_warped[:h_min, :w_min]


#     # get alpha masks
#     alpha1 = get_alpha_mask(img1_warped)
#     alpha2 = get_alpha_mask(img2_warped)

#     # weighted blending
#     mosaic = (img1_warped.astype(np.float32) * alpha1 + img2_warped.astype(np.float32) * alpha2) / (alpha1 + alpha2 + 1e-6) # debug for zerodiv error
#     return np.clip(mosaic, 0, 255).astype(np.uint8)

## ahhhhhh need to restart this is so confusing

# def make_mask(img):
#     if img.ndim == 2:
#         mask=(img>0).astype(np.float32)
#     else:
#         mask = (img.sum(axis=2) > 0).astype(np.float32)
#     return mask



def stitch_and_blend(img1, img2, H):
    # Blend img1 onto img2 using homography H (img1 → img2).
    #ensure both are placed in the same translated mosaic coordinate frame.

    # get img1 and 2 heigh and widths
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # calculate corners of img1 and img2 in homogenous coords, (0,0) is top left coord
    x1 = np.array([0, w1, w1, 0], dtype=np.float32)
    y1 = np.array([0,0,h1, h1], dtype=np.float32)
    corners1 = np.vstack([x1, y1, np.ones_like(x1)]) #(3,4) debug

    x2 = np.array([0, w2, w2, 0], dtype=np.float32)
    y2 = np.array([0,0,h2, h2], dtype=np.float32)
    corners2 = np.vstack([x2, y2, np.ones_like(x2)]) #(3,4) debug

    print("H shape", H.shape)
    print("corners1 shape", corners1.shape)
    print("result shape", (H @ corners1).shape)

    # warp img1's corners to img2's coordinate space
    warped_corners1 = H @ corners1
    warped_corners1 /= warped_corners1[2, :] # normalize so width = 1

    # combine both sets of corners to get mosaic bounds
    all_x = np.hstack([warped_corners1[0,:], corners2[0,:]])
    all_y = np.hstack([warped_corners1[1,:], corners2[1,:]])

    # compute min ans max pixel coords of combined mosaic canvas
    min_x, max_x = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    min_y, max_y = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))

    # compute total mosaic canvas width and heigt
    out_w, out_h = max_x - min_x, max_y - min_y

    print(f"mosaic bounds are: width {out_w}, height {out_h}")

    # BUILDING TRANSLATION MATRIX must shift all data to positive pixel coords
    # if part of warped img has negative coords, translate to make top left corner (0,0)
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

    #transform both imgs to same mosaic frame
    H_img1_to_mosaic = T @ H  # warp img1 and translate to canvas
    H_img2_to_mosaic = T # only translate img2 to canvas

    # actually warp the imgs
    warped_img1 = warpImageBilinear(img1, H_img1_to_mosaic, out_w, out_h)
    warped_img2 = warpImageBilinear(img2, H_img2_to_mosaic, out_w, out_h)

    print("warped_img1.shape", warped_img1.shape, "warped_img2.shape", warped_img2.shape)


    # now can blend for smooth overlap... first create blending masks
    # binar masks for where each warped img has pixels
    # mask1 = make_mask(warped_img1)
    # mask2 = make_mask(warped_img2)

    # # find the overlap where both imgs have pixels
    # only1 = (mask1 > 0) & (mask2 == 0) # where only img1 pixels
    # only2 = (mask2 > 0) & (mask1 == 0) # where only img2 pixels
    # overlap = (mask1 > 0) & (mask2 > 0)

    # # strat w simple linear blend... facier for wieghted blending later
    # mosaic = np.zeros_like(warped_img2, dtype=np.float32)

    # # copy non overlapping pixels directly
    # mosaic[only1] = warped_img1[only1]
    # mosaic[only2] = warped_img2[only2]

    # # weighted bending
    # mosaic[overlap] = 0.5 * (warped_img1[overlap] + warped_img2[overlap])
    # mosaic =  np.clip(mosaic, 0, 255).astype(np.uint8)

    mask1 = (warped_img1.sum(axis=2) > 0).astype(np.float32)
    mask2 = (warped_img2.sum(axis=2) > 0).astype(np.float32)

    # feather masks for smoother blend
    from scipy.ndimage import distance_transform_edt
    feather1 = distance_transform_edt(mask1)
    feather2 = distance_transform_edt(mask2)

    # normalize to [0,1]
    feather1 = feather1 / (feather1.max() + 1e-6)
    feather2 = feather2 / (feather2.max() + 1e-6)

    w1 = feather1[..., None]
    w2 = feather2[..., None]
    mosaic = (warped_img1 * w1 + warped_img2 * w2) / (w1 + w2 + 1e-6)
    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)

    return mosaic



if __name__ == "__main__":
    img_pair_name = "A"  # A, B, or C

    img1 = cv2.cvtColor(cv2.imread(f"./data/{img_pair_name}1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(f"./data/{img_pair_name}2.jpg"), cv2.COLOR_BGR2RGB)
   
    with open(f"./data/{img_pair_name}_correspondences.json", "r") as f:
        correspondences = json.load(f)
    img1_pts, img2_pts = np.array(correspondences["im1Points"]), np.array(correspondences["im2Points"])

    H = computeH(img2_pts, img1_pts)


    mosaic = stitch_and_blend(img1, img2, H)

    # visualize mosaic
    plt.figure(figsize=(10, 8))
    plt.imshow(mosaic)
    plt.title(f"Final Mosaic ({img_pair_name})")
    plt.axis("off")
    plt.show()

    # plot correspondences ---
    min_x, max_x, min_y, max_y = get_mosaic_bounds(img1, img2, H)
    translate = np.array([[1, 0, -min_x],
                          [0, 1, -min_y],
                          [0, 0, 1]], dtype=np.float64)
    H2 = translate @ H

    # transform correspondence points
    img2_pts_h = np.hstack([img2_pts, np.ones((len(img2_pts), 1))])
    warped_pts = (H2 @ img2_pts_h.T)
    warped_pts /= warped_pts[2]
    warped_pts = warped_pts[:2].T
    img1_pts_mosaic = img1_pts + np.array([-min_x, -min_y])

    plt.figure(figsize=(10, 8))
    plt.imshow(mosaic)
    plt.scatter(img1_pts_mosaic[:, 0], img1_pts_mosaic[:, 1], color='red', s=30, label='img1_pts')
    plt.scatter(warped_pts[:, 0], warped_pts[:, 1], color='blue', s=30, label='img2 warped')
    plt.legend()
    plt.title("Correspondence alignment on mosaic canvas")
    plt.show()