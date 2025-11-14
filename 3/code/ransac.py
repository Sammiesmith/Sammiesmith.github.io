import numpy as np

def compute_homography(points1, points2):
    # compute homography transformation from points1 to points 2
    points1 = points1[:, ::-1].astype(np.float32) # (x,y)
    points2 = points2[:, ::-1].astype(np.float32) # (x,y)

    assert points1.shape == points2.shape
    num_points = points1.shape[0]
    A = []

    # build matrix A 
    for i in range(num_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1 *x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])

    A = np.asarray(A, dtype=np.float32)
    # Solve Ah = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[2, 2] # return normalized homography (st last element = 1)

def transform_points(H, points):
    # apply homography H to set of N (x,y) points. points has shape (N,2)
    points = points[:, ::-1]
    points_with_homog_coord = np.hstack([points, np.ones((points.shape[0], 1))]) # (N, 3)
    projected_points = (H @ points_with_homog_coord.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3] # ensure homog coord = 1
    return projected_points[:, ::-1] # back to (y,x)



def ransac(points1, points2, num_iter=2000, threshold=3.0):
    best_H = None
    best_inlier_count = 0
    best_inliers = None
    N = len(points1)

    for iteration in range(num_iter):
        # randomly pick 4 correspondences
        sample_idxs = np.random.choice(N, 4)
        sample1, sample2 = points1[sample_idxs], points2[sample_idxs] # randomply sample 2 correspondences

        # Fit homography from sample
        H = compute_homography(sample1, sample2) # get the transformation between correspondences

        # project all pts1 using H and compute reprojection error
        projected = transform_points(H, points1)
        errors = np.linalg.norm(projected - points2, axis=1)

        # Count inliers using the fitted model
        inliers = errors < threshold
        inlier_count = np.sum(inliers)

        # Keep best model (the one with the most inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_H = H
    
    # Refit using all inliers
    if best_inliers is not None and np.sum(best_inliers) >= 4: # need at least 4 inlier correspondences
        best_H = compute_homography(points1[best_inliers], points2[best_inliers])

    return best_H, best_inliers



