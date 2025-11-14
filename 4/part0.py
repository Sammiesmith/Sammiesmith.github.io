# Part 0: Calibrating Your Camera and Capturing a 3D scan

# Part 0.1 Calibrating your camera
# step 1) take 30-50 pics w same zoom level on a calibration tag. 
# step 2) write a script to calibrate camera using the images. pipeline:
    # loop thru calibration imgs
    # for each img, detect the ArUco tags using opencv's aruco detector
    # extract corner coords from the detected tags
    # collect all detected corners and corresponding 3D world coords
        #( consider aruco tag as the world origin. 4 corners of tag's 3d points relative to that origin)
        # , e.g., if your tag is 0.02m × 0.02m, the corners could be [(0,0,0), (0.02,0,0), (0.02,0.02,0), (0,0.02,0)])
    # use cv2.calibrateCamera() to compute camera intrinsics and distortion effects

# code must handle cases where tags arent detected in some imgs. 

import cv2
import numpy as np
import os
import glob

# configuration ------------------------------------------------------

# Create ArUco dictionary and detector parameters (4x4 tags)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

tag_size = 0.02 # meters

# detect markers in a single img
def detect_aruco_corners(image):
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    if ids is None:
        return [],[]
    return corners, ids.flatten()

# generate 3d coords of the tag corners, assuming each tag is the same size
def get_tag_corner_coords(tag_size=tag_size):
    return np.array([
        [0,0,0], # top left
        [tag_size, 0, 0],  # topright
        [tag_size, tag_size, 0], # bottom right
        [0, tag_size, 0] # bottom left
    ], dtype=np.float32)

def get_tag_corner_coords_multi(tag_id, tag_size=0.06, tag_spacing=0.015):
    """
    Generate 3D coords for a specific tag ID based on a 3x2 grid layout.
    
    Args:
        tag_id: The ArUco tag ID (0-5)
        tag_size: Size of each tag in meters (0.06m)
        tag_spacing: Gap between tags in meters (0.015m)
    """
    # 3x2 grid: 3 columns, 2 rows
    row = tag_id // 3
    col = tag_id % 3
    
    # Distance between tag centers
    center_spacing = tag_size + tag_spacing
    
    # Origin of this tag (bottom-left corner)
    x_offset = col * center_spacing
    y_offset = row * center_spacing
    
    return np.array([
        [x_offset, y_offset, 0],
        [x_offset + tag_size, y_offset, 0],
        [x_offset + tag_size, y_offset + tag_size, 0],
        [x_offset, y_offset + tag_size, 0]
    ], dtype=np.float32)

def get_img_paths(image_folder, file_type="*.jpeg"):
    return sorted(glob.glob(os.path.join(os.path.dirname(__file__),image_folder, "*.jpeg")))


# citation: gpt generated img resizer
def resize_images(img_paths, output_folder="scan_photos_resized", size=(200, 200)):
    """
    Resize all JPEGs in img_paths to the given size and save to output_folder.

    Args:
        img_paths (list[str]): List of file paths to images.
        output_folder (str): Folder to save resized images.
        size (tuple[int, int]): Target size (width, height).
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving resized images to: {os.path.abspath(output_folder)}")

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {path}")
            continue

        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, resized)
        print(f"✅ Saved {filename} ({size[0]}x{size[1]})")

    print("🎉 Done resizing all images!")

# # get the calibration correspondences from all imgs
# def collect_calibration_points(image_folder, tag_size=tag_size):
#     object_pts = [] #3d pts in world coords
#     img_pts = [] # 2d pts in image plane

#     tag_corners = get_tag_corner_coords(tag_size)
#     img_paths = get_img_paths(image_folder)

#     # DEBUG
#     # print(f"Fount {len(img_paths)} imgs in {image_folder}")

#     for img_path in img_paths:
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         corners, ids = detect_aruco_corners(img)

#         if len(corners) == 0:
#             print(f"skipping img bc corners not detected {img_path}")
#             continue

#         for c in corners:
#             c = c.reshape(4,2)
#             img_pts.append(c.astype(np.float32))
#             object_pts.append(tag_corners.copy())
#         print(f"found {len(corners)}")
#     return object_pts, img_pts, img.shape[::-1]

def collect_calibration_points(image_folder, tag_size=tag_size):
    object_pts = []
    img_pts = []
    
    img_paths = get_img_paths(image_folder)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_aruco_corners(img)

        if len(corners) == 0:
            print(f"skipping img bc corners not detected {img_path}")
            continue

        # Now handle each detected tag with its proper ID
        for i, tag_id in enumerate(ids):
            c = corners[i].reshape(4, 2)
            img_pts.append(c.astype(np.float32))
            
            # Get the correct 3D coordinates for this specific tag
            tag_corners = get_tag_corner_coords_multi(tag_id, tag_size)
            object_pts.append(tag_corners)
            
        print(f"found {len(corners)} tags in {os.path.basename(img_path)}")
        
    return object_pts, img_pts, img.shape[::-1]

# run final camera calibration
def calibrate_camera_from_aruco(image_folder, tag_size=tag_size):
    print("Collecting Calibration corners")
    obj_pts, img_pts, img_size = collect_calibration_points(image_folder=image_folder, tag_size=tag_size)

    print("Calibrating...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    print(f"Camera Intrinsics matrix: {camera_matrix}")
    print(f"Distortion Coefficients: {dist_coeffs}")
    print(f"Reprojection Error: {ret}")
    return camera_matrix, dist_coeffs, rvecs, tvecs



##########################################################################################
# Part 0.3 estimating camera poses

def solve_camera_pose_for_img(img, camera_matrix, dist_coeffs, tag_size, reference_tag_id=0):
    # dectect aruco tag in img and solve PnP to get camera pose. return none if no tag found
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_aruco_corners(img)

    if len(corners) == 0:
        return None # skip img
    
    if reference_tag_id not in ids:
        return None # reference tag is not visible
    
    # take 1st detected tag
    idx = np.where(ids == reference_tag_id)[0][0]
    img_corners = corners[idx].reshape(4,1,2).astype(np.float32)
    obj_corners = get_tag_corner_coords(tag_size).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(obj_corners, img_corners, camera_matrix, dist_coeffs)

    if not success:
        return None
    return rvec, tvec

def extrinsics_world_to_camera(rvec, tvec):
    # convert solve Pnp output --> 3x4 world->camera matrix
    R, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack([R, tvec])
    return extrinsic

def invert_extrinsic_to_c2w(extrinsic):
    # convert world-> camera --> camera->world for nerf
    # fast version from disucssion rather than just do the invs
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_inv
    c2w[:3, 3] = t_inv
    return c2w

def estimate_poses(img_folder, camera_matrix, dist_coeffs, tag_size):
    # loop over all of the imgs from the object scan and compute the camera poses
    # return c2ws and imgs
    img_paths = get_img_paths(img_folder)
    c2ws = []
    imgs = []

    for path in img_paths:
        img = cv2.imread(path)
        result = solve_camera_pose_for_img(img, camera_matrix, dist_coeffs, tag_size)
        
        if result is None:
            print(f"no tag detected for img {os.path.basename(path)}")
            continue
        
        rvec, tvec = result
        w2c = extrinsics_world_to_camera(rvec, tvec)
        c2w = invert_extrinsic_to_c2w(w2c)

        c2ws.append(c2w)
        imgs.append(img)
        print(f"pose estimated for img {os.path.basename(path)}")
    
    return c2ws, imgs

import viser 
import time

def visualize_in_viser(c2ws, imgs, K):
    # visualize camera frustums in viser
    server = viser.ViserServer(share=False)
    H,W = imgs[0].shape[:2]
    f = K[0,0] # focal length

    for i, (c2w, img) in enumerate(zip(c2ws, imgs)):
        name = f"camera_{i}"
        server.scene.add_camera_frustum(
            name, 
            fov=2 * np.arctan2(H/2, f),
            aspect=W/H,
            scale=0.05,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3,3],
            image=img[..., ::-1] # bgr -> rgb
            )
    while True:
        time.sleep(0.1)


###########################################################################################
# Part 0.4 undistorting imgs

def undistort_imgs(imgs, K, dist_coeffs, lafufu=False):
    undistorted = []
    h,w = imgs[0].shape[:2]
    
    # deal w black borders
    alpha = 0 # crop ALL black borders
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), alpha, (w,h))
    x,y,w_roi,h_roi = roi
    for img in imgs:
        if not lafufu:
            undistorted_img = cv2.undistort(img, K, dist_coeffs, None, new_K)
            # crop to the valid region
            undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]
            undistorted.append(undistorted_img)
        else:
            undistorted_img = cv2.undistort(img, K, dist_coeffs)
            undistorted.append(undistorted_img)

    if not lafufu:
        # update principle pt for cropped coord system
        new_K[0,2] -= x
        new_K[1,2] -= y
    return np.array(undistorted), new_K

import matplotlib.pyplot as plt
# citation: gpt generated fn to allow me to inspect the undistorted imgs using matplotlib
def show_sample_images(imgs, n=5, title="Undistorted images"):
    """
    Display the first n images from a list using matplotlib.
    Works in Colab or locally.
    """
    n = min(n, len(imgs))
    plt.figure(figsize=(15, 3))
    for i in range(n):
        img_rgb = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i + 1)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"{title}\n{i}")
    plt.show()

def split_dataset(imgs, c2ws, train_proportion=0.7, val_proportion=0.15):
    N = len(imgs)
    N_train = int(N * train_proportion)
    N_val = int(N * val_proportion)

    train_imgs = imgs[:N_train]
    train_c2ws = c2ws[:N_train]

    val_imgs = imgs[N_train : N_train + N_val]
    val_c2ws = c2ws[N_train : N_train + N_val]

    test_imgs = imgs[N_train + N_val:]
    test_c2ws = c2ws[N_train + N_val:]

    return train_imgs, train_c2ws, val_imgs, val_c2ws, test_imgs, test_c2ws

def get_focal_len(K):
    return float(K[0,0])

def save_data_to_npz(train_imgs, train_c2ws, val_imgs, val_c2ws, test_imgs, test_c2ws, focal, out_path="my_data.npz"):
    np.savez(
        out_path,
        images_train = train_imgs,
        c2ws_train = train_c2ws,
        images_val = val_imgs,
        c2ws_val = val_c2ws,
        c2ws_test = test_c2ws,
        focal=focal
    )
    print(f"Saved data to {out_path}")



if __name__ == "__main__":

    img_paths = get_img_paths("scan_photos")
    resize_images(img_paths, output_folder="scan_photos_resized", size=(200,200))
    img_paths = get_img_paths("calibration_photos")
    resize_images(img_paths, output_folder="calibration_photos_resized", size=(200,200))
    
    lafufu = False
    tag_size = 0.06 # meters

    calib_imgs = "calibration_photos_resized"
    obj_scan_imgs = "scan_photos_resized"
    out_path = "my_data.npz"


    # calib_imgs = "lafufu_calibration_photos"
    # obj_scan_imgs = "lafufu_scan_photos"
    # out_path = "lafufu_data.npz"
    K, dist_coeffs, rvects, tvects = calibrate_camera_from_aruco(calib_imgs, tag_size)
    print("==============================================================")
   
    c2ws, imgs = estimate_poses(obj_scan_imgs, K, dist_coeffs, tag_size)
    # visualize_in_viser(c2ws, imgs, K)

    print("undistorting imgs")
    undistorted_imgs, new_K = undistort_imgs(imgs, K, dist_coeffs, lafufu=lafufu)

    print("showing some undistorted images...")
    show_sample_images(undistorted_imgs, n=5)

    print("splitting train, val, test data")
    images_train, c2ws_train, images_val, c2ws_val, images_test, c2ws_test = split_dataset(undistorted_imgs, np.array(c2ws))
    if lafufu:
        focal = get_focal_len(new_K)
    else:
        focal = get_focal_len(K)

    print("saving to npz")
    save_data_to_npz(
        images_train, c2ws_train,
        images_val, c2ws_val,
        images_test, c2ws_test,
        focal,
        out_path=out_path
    )
   

