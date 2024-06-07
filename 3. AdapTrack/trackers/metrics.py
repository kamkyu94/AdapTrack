import numpy as np
from scipy.spatial.distance import cdist


def iou(boxes_a, boxes_b):
    # Calculate area of each box
    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]

    # tlwh to tlbr
    boxes_a_tl, boxes_a_br = boxes_a[:, :2], boxes_a[:, :2] + boxes_a[:, 2:]
    boxes_b_tl, boxes_b_br = boxes_b[:, :2], boxes_b[:, :2] + boxes_b[:, 2:]

    # Calculate area of intersection
    top_left = np.maximum(boxes_a_tl[:, None, :], boxes_b_tl[None, :, :])
    bottom_right = np.minimum(boxes_a_br[:, None, :], boxes_b_br[None, :, :])
    area_intersection = np.maximum(0, bottom_right - top_left).prod(axis=2)

    # Return IoU
    return area_intersection / (area_a[:, None] + area_b[None, :] - area_intersection)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    # Check
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # Initialize
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    candidates = np.asarray([detections[i].tlwh for i in detection_indices])

    # Run
    for row, track_idx in enumerate(track_indices):
        # Disconnect lost tracks
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row] = 1.
            continue

        # Get bbox, Calculate IoU cost matrix
        bbox = tracks[track_idx].to_tlwh()
        cost_matrix[row] = 1. - iou(bbox[None], candidates)

    return cost_matrix, np.min(cost_matrix), 1.


def iou_constraint(tracks, detections, track_indices=None, detection_indices=None):
    # Initialize
    bboxes = np.asarray([tracks[i].to_tlwh() for i in track_indices])
    candidates = np.asarray([detections[i].tlwh for i in detection_indices])

    # Calculate cost matrix (IoU distance)
    cost_matrix = 1. - iou(bboxes, candidates)

    return cost_matrix


class NearestNeighborDistanceMetric(object):
    def __init__(self):
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        # Select sample to compare
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            self.samples[target] = self.samples[target][-1:]

        # Select sample to compare
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        # Initialize
        target_features = np.concatenate([self.samples[target] for target in targets], axis=0)

        # Calculate cost matrix (cosine distance)
        cost_matrix = cdist(target_features, features, metric='cosine')

        return cost_matrix
