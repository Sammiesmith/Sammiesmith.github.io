import numpy as np
import json
# look at discussion for homography matrix building

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    scale = np.sqrt(2) / std
    T = np.array([[scale, 0, -scale*mean[0]],
                  [0, scale, -scale*mean[1]],
                  [0, 0, 1]])
    pts_h = np.hstack([pts, np.ones((pts.shape[0],1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


# def computeH(img1_pts, img2_pts):
#     # img1_pts & img2_pts are an (N, 2) array with N points each with x,y coords
#     # return 3x3 homography array

#     assert (len(img1_pts) == len(img2_pts)) & (img1_pts.shape[1] == img2_pts.shape[1]) # same number of reference points in both imgs
#     num_correspondences = len(img1_pts)
#     assert num_correspondences >= 4 # need at least 4 correspondences for 8 degrees of freedom in a homography

#     pts1_norm, T1 = normalize_points(img1_pts)
#     pts2_norm, T2 = normalize_points(img2_pts)

#     # find A that solves Ah = b where h [h1 h2 ... h8].T and where b = [u, v] 
#     # each correspondence gets two equations (like in discussion)
#     # [ x y 1 0 0 0 -ux -uy] * h = u
#     # [ 0 0 0 x y 1 -vx -vy] * h = v

#     A = []
#     b = []

#     # calculate 2 rows of A for each correspondence; populate corresponding 2 rows of b
#     for (x, y), (u, v) in zip(pts1_norm, pts2_norm):
#         A += [[x, y, 1, 0, 0, 0, -u*x, -u*y]] # for u
#         A += [[0, 0, 0, x, y, 1, -v*x, -v*y]] # for v
#         b += [u]
#         b += [v]

#     A = np.array(A) # (2N, 8)
#     b = np.array(b) # (2N, 1)

#     print(f"System of equations: Ah = b \n A = {A} \n b = {b}")
#     print(f"Shape of A: {A.shape}")
#     print(f"Shape of b: {b.shape}")

#     # # Solve Ah = b using least squares (choose homography w lowest error)
#     # h, residuals, rank, singluar_vals = np.linalg.lstsq(A, b, rcond=None)

#     # # make homography matrix 3x3
#     # H = np.append(h, 1).reshape(3,3) # need to append 1 for scaling coef
#     # H /= H[2, 2] # need to normalize so bottom right entry is 1

#     # Solve instead via SVD (least-squares)
#     # Find h as the last column of V (smallest singular value)
#     U, S, Vt = np.linalg.svd(A)
#     h = Vt[-1, :]
#     H_norm = np.append(h,1).reshape(3, 3)

#     # Denormalize (undo the coordinate transforms) 
#     # H maps pts1→pts2 in normalized space, so we undo both transformations:
#     H = np.linalg.inv(T2) @ H_norm @ T1

#     # === 5. Scale so that bottom-right entry = 1 ===
#     H /= H[2, 2]

#     print(f"h = {H}")

#     return H

def computeH(img1_pts, img2_pts):
    """
    Computes 3x3 homography that maps img1_pts → img2_pts
    using the normalized DLT (Direct Linear Transform) algorithm.
    """
    assert (len(img1_pts) == len(img2_pts)) and (img1_pts.shape[1] == 2)
    assert len(img1_pts) >= 4

    # === 1. Normalize both point sets ===
    pts1_norm, T1 = normalize_points(img1_pts)
    pts2_norm, T2 = normalize_points(img2_pts)

    # === 2. Build the A matrix (2N × 9) ===
    A = []
    for (x, y), (u, v) in zip(pts1_norm, pts2_norm):
        A.append([-x, -y, -1,  0,  0,  0, u*x, u*y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)

    # === 3. Solve Ah = 0 using SVD ===
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H_norm = h.reshape(3, 3)

    # === 4. Denormalize ===
    H = np.linalg.inv(T2) @ H_norm @ T1

    # === 5. Normalize so H[2,2] = 1 ===
    H /= H[2, 2]
    return H



if __name__ == "__main__":
    img_pair_name = 'C' # adjust this to be A, B, or C

    with open("./data/" + img_pair_name + "_correspondences.json", "r") as f: # correspondence files must be in the format: [name]_correspondences.json
        correspondences = json.load(f)
        
    img1_pts = np.array(correspondences["im1Points"])
    img2_pts = np.array(correspondences["im2Points"])


    H = computeH(img1_pts, img2_pts)

    # double check that img1 * H = img2 for the first pt
    img1 = np.array([img1_pts[0][0], img1_pts[0][1], 1])
    img2_mapped = H @ img1
    img2_mapped /= img2_mapped[2]
    print(f"Expected point: {img2_pts[0]}, Mapped point: {img2_mapped[:2]}")

    print(f"Image Pair Name: {img_pair_name}")

    # Check reprojection error
    img1_h = np.hstack([img1_pts, np.ones((len(img1_pts),1))])
    proj = (H @ img1_h.T).T
    proj /= proj[:,2][:,None]
    err = np.linalg.norm(proj[:,:2] - img2_pts, axis=1)
    print("Mean reprojection error (px):", np.mean(err))






