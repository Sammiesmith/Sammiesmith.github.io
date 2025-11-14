import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def match_features(descriptors_img1, descriptors_img2, coords1, coords2, ratio_threshold=0.6):
    """
    goal= match feature descriptors between two images using Lowe's ratio test... as described in MOPS
    descriptors for img1 and img2= note: each row = flattened 8x8 patch
    coords1,2 = coordinates of features in image 1 and img 2
    ratio_threshold = Lowe's ratio threshold btwn 1-NN and 2-NN distances
    """
    matches = [] # list to hold (index in image1, index in image2)
    scores = [] # list to hold the distance of each accepted match

    # loop thru each descriptor in image 1
    for i, descriptor1 in enumerate(descriptors_img1):
        # compute distances from this descriptor to all descriptors in img 2
        # measures similarity betwn local patch around feature in image1 and 2 
        distances = np.linalg.norm(descriptors_img2 - descriptor1, axis=1) # euclidean distance

        #find best and second best matches
        # sort all dists in ascending order
        sorted_indices = np.argsort(distances)
        best_idx = sorted_indices[0] # idx of smallest distance (1NN)
        second_best_idx = sorted_indices[1] # idx of second smallest distance (2NN)

        best_dist = distances[best_idx]
        second_best_dist = distances[second_best_idx]

        # apply lowe's ratio test to see how much better the best match is than the second best
        # if they are close (w ratio close to 1), then the feature is ambiguous and we discard it
        ratio = best_dist / (second_best_dist + 1e-13)

        if ratio < ratio_threshold:
            # feature is useful, so we keep it
            matches.append((i, best_idx))
            scores.append(best_dist)

    matches = np.array(matches)
    scores = np.array(scores)

    return matches, scores


