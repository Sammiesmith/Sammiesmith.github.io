import numpy as np

# module that calculates adaptive_non_maximal_suppression according MOPS paper

def adaptive_non_maximal_suppression(coords, harris_corner_strength_map, c_robust=0.9, num_interest_pts=500):
    """
    coords = array of corner corrds from get_harris_corners
    harris_corner_strength_map = map of strengths of corners (shape shape as img)
    num_interest_pts = number of points to keep
    c_robust = supression robustness factor

    return anms_coords, radii

    anms_coords = array after corner coords after ANMS, sort by supression radius (strongest->weakest)
    radii = array of corresponding supression radii
    """
    y, x = coords 
    num_corners = len(y)

    corner_strengths = harris_corner_strength_map[y, x]

    radii = np.full(num_corners, np.inf) # initialize as infinity per research paper

    # calc supression radius for each corner
    for i in range(num_corners):
        # find points with a higher response by factor c_robust
        stronger_index = np.where(corner_strengths > corner_strengths[i] * c_robust)[0]
        if len(stronger_index) > 0:
            # if there are points with higher corner strength by factor c_robus
            # then compute the distances to these stronger points
            dy = y[i] - y[stronger_index]
            dx = x[i] - x[stronger_index]
            distances = np.sqrt(dx**2 + dy**2)
            radii[i] = np.min(distances) # new radius is the minimum distance to a stronger corner

    # pick num_interest_points that have the largest radii
    sorted_index = np.argsort(-radii)
    interest_pts = sorted_index[:num_interest_pts]

    anms_coords = np.vstack((y[interest_pts], x[interest_pts]))
    return anms_coords, radii[interest_pts]