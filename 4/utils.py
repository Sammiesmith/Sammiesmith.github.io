import numpy as np
import torch
#from dataloader import load_lego_data


# convert np array to torch array
def np_to_torch(M):
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).float()
    return M

def add_homog_coord_to_batched_pts(pts):
    # convert pts to homog coords pts (N,2) --> pts (N,3)
    ones = torch.ones(pts.shape[0],1, dtype=pts.dtype,device=pts.device)
    pts = torch.cat([pts, ones], dim=-1) # (N,3)
    return pts



def single_pt_2_batch_size_1(pt):
    # need to handle a single point vs batched pts
    is_single_pt = (pt.ndim == 1)
    if is_single_pt:
        pt = pt.unsqueeze(0) # this adds a batch dimension (3,) --> (1,3)
    return pt



# camera to world coordinate conversion
def transform(c2w, x_c):
    """
    c2w: camera to world transformation matrix (4,4) or (N,4,4) for batched points
    x_c: camera coords (3,) or (4,) w homog coord, (N,3) or (N,4) w homog coord for batched pts

    returns x_w: world coords (3,) or (N,3) depending on if input is single pt or batched pts

    useful equations:
    x_c = w2c @ x_w  
    ==> x_w = invs(w2c) @ x_c
    where invs(w2c) = c2w
    ==> we should compute x_w = c2w (4,4) @ x_c (4,)... 
    BUT bc in pytorch a vector is thought of as a row, so x_c really is (,4)
    x_w (4,) = c2w (4,4) @ x_c (,4) !!! mistmatch, so must transpose x_c
    x_w.T (4,) = c2w (4,4) @ x_c.T (4,) !!! we want (,4) (which is a column in pytorch)
    so we actually need to comput:
    x_w (,4) = x_c (,4) @ c2w.T (4,4)
    """
    # should convert inputs to torch arrrays to make upstream tasks w neural field easier
    x_c = np_to_torch(x_c)
    c2w = np_to_torch(c2w)

    # need to handle a single point vs batched pts
    # if x_c is a single pt, add batch dimenstion (3,) -- (1,3)
    is_single_pt = (x_c.ndim == 1)
    x_c = single_pt_2_batch_size_1(x_c)

    # convert to homog coords if needed
    if x_c.shape[-1] == 3:
        # x_c shape: (N,3) --> (N,4)
        x_c = add_homog_coord_to_batched_pts(x_c)

    # handle single c2w vs batch of c2ws
    if c2w.ndim == 2:
        # then we're dealing w a transformation matrix for a single pt (4,4)... need to apply to all pts
        # x_c shape (N, 4) ==> we get (N,4) = (N,4) = (N,4) @ (4,4).T
        x_w = x_c @ c2w.T 
    else:
        # we have a batch of transform matricies: (N, 4, 4)
        # x_c shape (N,4) ==> we get (N,4) = (N, 1,4) @ (N, 4, 4).T
        x_w = (x_c.unsqueeze(1) @ c2w.transpose(-2,-1)).squeeze(1)

    # now we need to remove the homog coord to get world coords
    x_w = x_w[..., :3] # (N,4) --> (N,3)

    # and should also remove batch dimension if it was a single pt
    if is_single_pt:
        x_w = x_w.squeeze(0) # (1,3) --> (3,)
    
    return x_w

# def test_transform_fn():
#     images_train, images_val, c2ws_train, c2ws_val, c2ws_test, focal = load_lego_data()
#     c2w = torch.from_numpy(c2ws_train[0]).float()
#     x_c = torch.randn(3)
#     w2c = torch.linalg.inv(c2w)
#     x_c_reconstructed = transform(w2c, transform(c2w, x_c))
#     print(f"Original camera coords: {x_c}, Transformed and Untransformed camera coords: {x_c_reconstructed}")



# pixel to camera coordinate conversion.
def pixel_to_camera(K, uv, s):
    """
    Consider a pinhole camera w 
    - focal length (fx, fx) 
    - principle point (ox = image_width/2, oy=image_height/2)
    - intrinsic matrix 
        K = [[fx, 0, ox],
             [0, fy, oy],
             [0, 0,  1]]
    
    To project a 3D point in x_c in camera coordinate system to 2D location in pixel coord system do:
        s (scalar) * uv (w homog coord)(3,) = K (3,3) @ x_c (3,)
        aka: s @ [u, v, 1].T = K @ x_c
        where s = z_c depth of the point along optical axis (aka z axis coord in x_c)

    Goal: solve for x_c

    Math:
    s * [u,v,1].T = K @ x_c
    ==> x_c = s * inv(K) @ [u,v,1].T

    Arugments:
    - K: (3,3)
    - uv: (2,) or (N,2) where N is batch size
    - s: () or (N) for batched pts
    Returns:
    - x_c (3,) or (N,)
    """
    # convert to torch if needed
    K = np_to_torch(K)
    uv = np_to_torch(uv)
    if isinstance(s, (int, float, np.ndarray)):
        s = torch.tensor(s, dtype=torch.float32)

    # handle single pt vs batch: uv (2,) -> uv (1,2)
    is_single_pt = (uv.ndim == 1)
    uv = single_pt_2_batch_size_1(uv)
    
    # handle scalar s vs batch of s
    if s.ndim == 0: # scalar
        s = s.unsqueeze(0) # () --> (1,2)
    if s.shape[0] == 1 and uv.shape[0] > 1:
        s = s.expand(uv.shape[0]) # need to broadcast to match batch size

    # convert uv to homog coords [u,v] (N,2) --> [u,v,1] (N,3)
    uv = add_homog_coord_to_batched_pts(uv)

    # compute K inverse
    K_inv = torch.linalg.inv(K) # (3,3)

    # apply formula: x_c = s * inv(K) @ uv_homog.T
    # rn uv_homog is (N,3), K_inv is (3,3)... we want x_c to be (N,3)
    # we should use x_c (N,3) = (uv_homog) @ K_inv.T to get right dims
    # x has shape (N,) need to brodcast t0 (N,1)
    s = s.unsqueeze(-1) # (N,) --> (N,1)
    x_c = uv @ K_inv.T
    x_c = s * x_c

    # remove batch dim if input was a single pt
    if is_single_pt:
        x_c = x_c.squeeze(0) # (1,3) --> (3,)
    
    return x_c

# pixel to ray.
def pixel_to_ray(K, c2w, uv):
    """
    Goal: convert a pixel coordinate into a ray with origina and normalized direction: ray_o, ray_d
    Math: 
        c2w = [[R (3,3), t (3,)],
                0, 0, 0, 1     ]]
    
        ray_o = t = c2w[:3, 3]
        ray_d = (x_w - ray_o) / (l2_norm(x_w, ray_o)) where x_w is (3,) world coords

    Arguments: 
    - K (3,3)
    - c2w (4,4)
    - uv (2,) or (N,2)
    Return:
    - ray_o (3,) or (N,3)
    - ray_d (3,) or (N,3)
    """
 
    K = np_to_torch(K)
    c2w = np_to_torch(c2w)
    uv = np_to_torch(uv)
    is_single_pt = (uv.ndim == 1)
    uv = single_pt_2_batch_size_1(uv) # (2,) -> (1,2)
    batch_size = uv.shape[0]

    s=torch.ones(batch_size, dtype=torch.float32, device=uv.device) # depth=1 for all pixels in batch

    # get ray origin (translation part)
    ray_o = c2w[:3, 3] # shape (3,)
    ray_o = ray_o.unsqueeze(0).expand(batch_size,3) # brodcast to match batchsize if needed (3,) -> (N,3)

    # get direction in camera space at depth s=1
    x_c = pixel_to_camera(K, uv, s) # (N,3)

    # transform x_c to world space
    # x_w = transform(c2w, x_c) # gets us point along ray in world coords (N,3)
    
    # compute direction vector ray_d
    # ray_d = x_w - ray_o #(N,3)
    ray_d = x_c @ c2w[:3, :3].T # only use the rotation part
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True) # (N,3)

    # must remove batch dim if input was single pt
    if is_single_pt:
        ray_o = ray_o.squeeze(0) #(1,3) -> (3,)
        ray_d = ray_d.squeeze(0) #(1,3) -< (3,)
    
    return ray_o, ray_d


# precompute all rays from all images
def create_rays_dataset(images, K, c2ws):
    """
    Aruguments:
    - images: training images (N_images, H, W, 3)
    - K: camera intrinsic matrix or tensor (3, 3)
    - c2ws: cam to world matricies shape (N_imgs, 4, 4)

    Returns:
        rays_o: all ray origins (N_images*H*W, 3)
        rays_d: all ray directions (N_images*H*W, 3)
        colors: all pixel colors (N_images*H*W, 3)
        uvs: all uv coords (N_images*H*W, 3)
    """
    N_images, H, W, _ = images.shape
    print(f"Creating dataset of rays----")
    print(f"Num imgs: {N_images}, Img size: {H},{W}")

    K = np_to_torch(K)

    # create pixel coord grid for one img... add 0.5 to get pixel centers
    u_coords = torch.arange(W, dtype=torch.float32) + 0.5
    v_coords = torch.arange(H, dtype=torch.float32) + 0.5

    # create the meshgrid and then flatten to get all pixel coords from the img
    u_grid, v_grid = torch.meshgrid(v_coords, u_coords, indexing='xy')
    uvs = torch.stack([u_grid.flatten(), v_grid.flatten()], dim=-1) # (H*W, 2)

    # compute rays for all imgs
    all_rays_o = []
    all_rays_d = []
    all_colors = []

    for i in range(N_images):
        # get the camera metrix for this img
        c2w = torch.from_numpy(c2ws[i]).float()
        # compute rays for this img
        rays_o, rays_d = pixel_to_ray(K, c2w, uvs) # (H*W, 3) for both
        # get colors for all pixels in this img
        colors = torch.from_numpy(images[i]).float().reshape(-1,3) # (H*W, 3)
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_colors.append(colors)

    # now we need to stakc all the rays into single tensors along N_img dim
    rays_o = torch.cat(all_rays_o, dim=0) # (N_imgs*H*W, 3)
    rays_d = torch.cat(all_rays_d, dim=0) # (N_imgs*H*W, 3)
    colors = torch.cat(all_colors, dim=0) # (N_imgs*H*W, 3)

    uvs_all = uvs.repeat(N_images, 1) # (N_imgs*H*W, 2) # DEBUG: save uvs for testing later
    print(f"Expected rays_o, rays_d, and colors shape: ({N_images*H*W}, 3)")
    print(f"Actual shapes: {rays_o.shape}, {rays_d.shape}, {colors.shape}")
    return rays_o, rays_d, colors, uvs_all

# randomly sample rays from precomputed rays dataset
def sample_rays(rays_o, rays_d, colors, num_samples):
    """
    args:
    - rays_o (N_rays, 3) N_rays = N_images*H*W
    - rays_d (N_rays, 3)
    - colors: (N_rays, 3)
    - num_samples: number of rays to sample (scalar)

    return:
        sampled_rays_o (num_samples, 3)
        sampled_rays_d (num_samples, 3)
        sampled_colors (num_samples, 3)
    """
    N_rays = rays_o.shape[0]
    idxs = torch.randperm(N_rays)[:num_samples]#torch.randint(0, N_rays, (num_samples,)) # w/o repl
    sample_rays_o = rays_o[idxs]
    sample_rays_d = rays_d[idxs]
    sample_colors = colors[idxs]
    return sample_rays_o, sample_rays_d, sample_colors


# sample points along rays to use for volume rendering
def sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=True):
    """
    samples n times along each ray btwn near and far dists
    can add random perturnbation to avoid overfitting (for training only)

    args:
    - rays_o (N_rays, 3) N_rays = N_images*H*W
    - rays_d (N_rays, 3)
    - near = bound for nearest sample
    - far = bound for farthest sample
    - n_samples: number of rays to sample (scalar)
    - perturb: boolean (true=add random perturbation to sampling dists).. true during training, false durign inference/eval

    returns:
    points: 3d sample pts (N_rays, n_samples, 3)
    t_vals: dists of each sample pts (N_rays, n_samples)

    math:
    point = ray_o + t * ray_d, where t sampled from [near, far]
    """
    N_rays = rays_o.shape[0]
    # uniform samples along [near, far]
    t_vals = torch.linspace(near, far, n_samples, dtype=rays_o.dtype, device=rays_o.device)
    # expand to match batch size (n_samples,) -> (N_rays, n_samples)
    t_vals = t_vals.unsqueeze(0).expand(N_rays, n_samples)
    # add perturbation for training
    if perturb:
        bin_width = (far-near) / n_samples
        t_rand = torch.rand_like(t_vals) * bin_width # random offset for each tval
        t_vals = t_vals + t_rand

    # finally can compute 3d pts along rays
    # first reshape for broadcasting
    rays_o = rays_o.unsqueeze(1) #(N_rays, 1, 3)
    rays_d = rays_d.unsqueeze(1) #(N_rays, 1, 3)
    t_vals_expanded = t_vals.unsqueeze(-1) # (N_rays, n_samples,1)

    points = rays_o + t_vals_expanded * rays_d # (N_rays, n_samples, 3)
    return points, t_vals






if __name__ == "__main__":
    test_transform_fn()










