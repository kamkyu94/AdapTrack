import numpy as np
from opts import opt
from sklearn.mixture import GaussianMixture
import trackers.kalman_filter as kalman_filter
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering


def gate_cost_matrix(cost_matrix, tracks, detections, track_indices, detection_indices):
    # Get gating threshold
    gating_threshold = kalman_filter.chi2inv95[4]

    # Get measurements (cxcyah) of the detections
    measurements = np.asarray([detections[i].to_cxcyah() for i in detection_indices])

    # Run
    for row, track_idx in enumerate(track_indices):
        # Get track
        track = tracks[track_idx]

        # Calculate gating distance
        gating_distance = track.kf.gating_distance(track.mean, track.covariance, measurements)

        # Update cost matrix
        cost_matrix[row, gating_distance > gating_threshold] = 1e5
        cost_matrix[row] = opt.gating_lambda * cost_matrix[row] + (1 - opt.gating_lambda) * gating_distance

    return cost_matrix


def set_threshold(dists, ori_threshold, min_anchor, max_anchor):
    # # More Sampling with linear assignment (Do not use)
    # indices = linear_assignment(dists)
    # dists = dists[indices[0], indices[1]]

    # Prepare
    threshold = ori_threshold
    dists_1d = dists.reshape(-1, 1)
    dists_1d = dists_1d[dists_1d < max_anchor]
    dists_1d = dists_1d[min_anchor < dists_1d]

    if len(dists_1d) > 0:
        # Prepare
        dists_1d = list(dists_1d) + [min_anchor, max_anchor]
        dists_1d = np.array(dists_1d).reshape(-1, 1)

        # Select Clustering
        model = KMeans(n_clusters=2, init=np.array([[min_anchor], [max_anchor]]), n_init=1, random_state=10000)
        # model = AgglomerativeClustering(n_clusters=2, linkage='ward')
        # model = SpectralClustering(n_clusters=2, assign_labels='kmeans', random_state=10000)
        # model = GaussianMixture(n_components=2, means_init=np.array([[min_anchor], [max_anchor]]), random_state=10000)

        # Fit
        result = model.fit_predict(dists_1d)

        # Rare exception (Only occurs with Gaussian mixture clustering)
        if np.sum(result == 0) == 0 or np.sum(result == 1) == 0:
            return ori_threshold

        # Set threshold
        threshold = min(np.max(dists_1d[result == 0]), np.max(dists_1d[result == 1])) + 1e-5
        # threshold = max(np.min(dists_1d[result == 0]), np.min(dists_1d[result == 1])) - 1e-5
        # threshold = (np.max(dists_1d[result == 0]) + np.min(dists_1d[result == 1])) / 2
        # threshold = (np.mean(dists_1d[result == 0]) + np.mean(dists_1d[result == 1])) / 2

    return threshold


def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    # For sure
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # Nothing to match.
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    # Split
    distance_metric, constraint_metric, adap_flag = distance_metric

    # Calculate cost matrix 1
    cost_matrix, _, cost_matrix_max = distance_metric(tracks, detections, track_indices, detection_indices)

    # Apply filtering
    if constraint_metric is not None:
        constraint_matrix = constraint_metric(tracks, detections, track_indices, detection_indices)
        cost_matrix[constraint_matrix == 1] = 1

    # Adaptively set threshold
    if adap_flag:
        max_distance = set_threshold(cost_matrix, max_distance, 0., cost_matrix_max)

    # Adjust cost matrix
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # Hungarian algorithm
    indices = linear_assignment(cost_matrix)

    # Initialization
    matches, unmatched_tracks, unmatched_detections = [], [], []

    # Update matching results 1
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[0]:
            unmatched_tracks.append(track_idx)

    # Update matching results 2
    for row, col in np.concatenate([indices[0][:, None], indices[1][:, None]], axis=1):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections
