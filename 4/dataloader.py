import cv2
import numpy as np
import matplotlib.pyplot as plt



#####################################################################################
# Part 1 dataloader
####################################################################################

def read_img(path):
    img = cv2.imread(path) # reads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
    return img

def create_meshgrid(h, w):
    # return a (u,v) meshgrid of normalized pixel coordinates

    # create coord arrays
    u_coords = np.arange(w) # u = horizontal axis # [0,1,2,3]
    v_coords = np.arange(h) # v = vertical axis # [0,1,2]

    # create 2D grids
    u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing='xy')

    # Example if h=3, w=4:
    # u_grid = [[0,1,2,3],
    #           [0,1,2,3],
    #           [0,1,2,3]]
    #
    # v_grid = [[0,0,0,0],
    #           [1,1,1,1],
    #           [2,2,2,2]]

    # flatten and stack into (num_pixels, 2) array
    u_flat = u_grid.flatten() # [0,1,2,3,  0,1,2,3,  0,1,2,3]
    v_flat = v_grid.flatten()
    coords = np.stack([u_flat, v_flat], axis=-1).astype(np.float32)

    # Example:
    # [[0,0],
    #  [1,0],
    #  [2,0],
    #  [3,0],
    #  [0,1],
    #  ...
    # ]

    # normalize coords to [0,1]
    coords[:, 0] = coords[:,0] / w # normaalize u
    coords[:, 1] = coords[:, 1] / h # normalize v

    print(f"created coordinate grid of shape {coords.shape}")
    print(f"u range: [{coords[:, 0].min()}, {coords[:,0].max}]")
    print(f"v range: [{coords[:, 1].min()}, {coords[:,1].max}]")

    return coords

def normalize_color(img):
    # Flatten image colors and normalize color to [0,1]
    # input: img (h,w,3) RBG with values [0,255]
    # return: color array (h*w, 3) RGB with values [0,1]

    colors = img.reshape(-1, 3).astype(np.float32)
    colors = colors / 255.0

    print(f"flattened colors shape {colors.shape}")
    print(f"color range: [{colors.min(), colors.max()}]")
    return colors


def sample_pixels(coords, colors, N):
    # randomly grab N indices
    # use the indices to grab N coords and N colors
    # coords (num_pixels, 2), colors (num_pixels, 3), N num samples
    # return sampled_coords (num_samples, 2), sampled_colors (num_samples, 3)
    num_pixels = coords.shape[0]
    idxs = np.random.choice(num_pixels, size=N, replace=False)
    sampled_coords = coords[idxs]
    sampled_colors = colors[idxs]
    print(f"Sampled {N} coordinates (shape={sampled_coords.shape}) and colors (shape={sampled_colors.shape}) for training")
    return sampled_coords, sampled_colors



def visualize_samples(img, coords, colors, num_samples=100):
    # visualize random pixel samples to test the dataloader
    sampled_coords, sampled_colors = sample_pixels(coords, colors, num_samples)
    h,w = img.shape[:2]

    # convert normalized coords --> pixel coords
    u_pixels = sampled_coords[:, 0] * w
    v_pixels = sampled_coords[:, 1] * h 
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5))

    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.scatter(u_pixels, v_pixels, c=sampled_colors, s=1)
    ax2.set_xlim(0,w)
    ax2.set_ylim(h,0)
    ax2.set_aspect('equal')
    ax2.set_title(f'{num_samples} Random Samples')
    ax2.set_xlabel('u width')
    ax2.set_ylabel('v width')

    plt.tight_layout()
    plt.show()

def load_lego_data(data_path="lego_200x200.npz"):
    data = np.load(data_path)

    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images 
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images: 
    images_val = data["images_val"] / 255.0

    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]

    # Test cameras for novel-view video rendering: 
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]

    # Camera focal length
    focal = data["focal"]  # float
    return images_train, images_val, c2ws_train, c2ws_val, c2ws_test, focal

def load_my_data(data_path="my_data.npz"):
    data = np.load(data_path)

    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images 
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images: 
    images_val = data["images_val"] / 255.0

    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]

    # Test cameras for novel-view video rendering: 
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]

    # Camera focal length
    focal = data["focal"]  # float
    return images_train, images_val, c2ws_train, c2ws_val, c2ws_test, focal

#####################################################################################
# Part 2.3 Putting the Dataloader all together
####################################################################################

import viser, time  # pip install viser
import numpy as np
from utils import create_rays_dataset, sample_rays, sample_points_along_rays
import torch

# load data
images_train, _, c2ws_train, _, _, focal = load_lego_data()
H, W = images_train.shape[1:3]
focal = float(focal)

# intrinsics matrix
K = np.array([[focal, 0.0, W/2],
            [0.0, focal, H/2],
            [0.0, 0.0, 1.0]])

# create the rays dataset
rays_o_all, rays_d_all, colors_all, uvs_all = create_rays_dataset(images_train, K, c2ws_train)


def show_all_cameras_and_rays():
    # sample 100 rays
    rays_o, rays_d, pixels = sample_rays(rays_o_all, rays_d_all, colors_all, 100)

    # sample points along the rays
    points, t_vals = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=True)

    # convert to numpy bc viser requires numpy not torch arrays
    rays_o = rays_o.numpy() if isinstance(rays_o, torch.Tensor) else rays_o
    rays_d = rays_d.numpy() if isinstance(rays_d, torch.Tensor) else rays_d
    points = points.numpy() if isinstance(points, torch.Tensor) else points
    # ---------------------------------------

    server = viser.ViserServer(share=True)

    # add camera frustrums
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image
        )

    # add rays
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.add_spline_catmull_rom(
            f"/rays/{i}", positions=np.stack((o, o + d * 6.0)),
        )

    # add sample pts
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )

    print("Visualizeation ready, open url in browser.")

    # keep server running
    while True:
        time.sleep(0.1)  # Wait to allow visualization to run

def show_all_rays_for_one_camera():
    # check that uv indexing is working
    uvs_start = 0
    uvs_end = H*W
    sample_uvs = uvs_all[uvs_start:uvs_end]

    # convert to integer coords for indexing
    u_int = (sample_uvs[:,0] - 0.5).long().numpy()
    v_int = (sample_uvs[:,1] - 0.5).long().numpy()

    # check that colors match
    colors_from_uvs = colors_all[uvs_start:uvs_end].numpy()
    colors_from_image = images_train[0, v_int, u_int]
    assert np.allclose(colors_from_uvs, colors_from_image, atol=1e-5)
    print("uv indexing is correct")

    # sample rays from first img only
    indices = np.random.randint(low=0, high=H*W, size=100)

    # # Uncomment this to display random rays from the top left corner of the image
    # indices_x = np.random.randint(low=100, high=200, size=100)
    # indices_y = np.random.randint(low=0, high=100, size=100)
    # indices = indices_x + (indices_y * 200)

    rays_o = rays_o_all[indices]
    rays_d = rays_d_all[indices]

    # sample points along the rays
    points, t_vals = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, n_samples=32, perturb=True)

    # convert to numpy bc viser requires numpy not torch arrays
    rays_o = rays_o.numpy() if isinstance(rays_o, torch.Tensor) else rays_o
    rays_d = rays_d.numpy() if isinstance(rays_d, torch.Tensor) else rays_d
    points = points.numpy() if isinstance(points, torch.Tensor) else points

    #-------------------------------
    server = viser.ViserServer(share=True)

    
    server.scene.add_camera_frustum(name="first_training_img_from_lego_dataset",
        fov=2 * np.arctan2(H / 2, K[0, 0]),
        aspect=W / H,
        scale=0.15,
        wxyz=viser.transforms.SO3.from_matrix(c2ws_train[0][:3, :3]).wxyz,
        position=c2ws_train[0][:3, 3],
        image=images_train[0]
    )

    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        positions = np.stack((o, o + d * 6.0))
        server.scene.add_spline_catmull_rom(
            f"/rays/{i}", positions=positions,
        )

    server.scene.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.03,
    )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run




if __name__ == "__main__":
    # img_path = "C:/Users/sammi/Sammiesmith.github.io/4/fox.jpg"
    # image = read_img(img_path)
    # h,w = image.shape[:2]
    # coords = create_meshgrid(h,w)
    # colors = normalize_color(image)
    # visualize_samples(image, coords, colors, num_samples=100)

    # show_all_cameras_and_rays()
    show_all_rays_for_one_camera()
