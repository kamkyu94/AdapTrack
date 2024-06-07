from opts import opt
from trackers.cmc import *
from trackers import metrics
from trackers.units import Track
from trackers import linear_assignment


class Tracker:
    def __init__(self, metric, vid_name):
        # Set parameters
        self.metric = metric

        # Initialization
        self.tracks = []
        self.next_id = 1

        # Set camera motion compensation model
        self.cmc = CMC(vid_name)

    def initiate_track(self, detection):
        self.tracks.append(Track(detection.to_cxcyah(), self.next_id, detection.confidence, detection.feature))
        self.next_id += 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def camera_update(self):
        # Get warp matrix
        warp_matrix = self.cmc.get_warp_matrix()

        # Warp
        for track in self.tracks:
            apply_cmc(track, warp_matrix)

    def gated_metric(self, tracks, detections, track_indices, detection_indices):
        # Gather
        targets = np.array([tracks[i].track_id for i in track_indices])
        features = np.array([detections[i].feature for i in detection_indices])

        # Calculate cosine distances
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix_min = np.min(cost_matrix)
        cost_matrix_max = np.max(cost_matrix)

        # Gating
        cost_matrix = linear_assignment.gate_cost_matrix(cost_matrix, tracks, detections,
                                                         track_indices, detection_indices)

        return cost_matrix, cost_matrix_min, cost_matrix_max

    # Match
    def match(self, detections):
        # Split tracks into confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks and high-confident detections using appearance features
        # Give [self.gated_metric, metrics.iou_constraint, True] to turn on adap thresholding
        # Give [self.gated_metric, metrics.iou_constraint, False] to turn off adap thresholding
        matches_a, _, unmatched_detections = \
            linear_assignment.min_cost_matching([self.gated_metric, metrics.iou_constraint, True],
                                                opt.max_distance, self.tracks,
                                                detections, confirmed_tracks)

        # Gather unmatched tracks
        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _ in matches_a))

        # Gather (remaining tracks + unconfirmed track), lost tracks
        candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        # Associate (remaining tracks + unconfirmed tracks) and remaining detections using IoU
        # Give [metrics.iou_cost, None, True] to turn on adap thresholding
        # Give [metrics.iou_cost, None, False] to turn off adap thresholding
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching([metrics.iou_cost, None, True], opt.max_iou_distance, self.tracks,
                                                detections, candidates, unmatched_detections)

        # Update
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections):
        # Run match
        matches, unmatched_tracks, unmatched_detections = self.match(detections)

        # Update tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            if detections[detection_idx].confidence >= opt.conf_thresh:
                self.initiate_track(detections[detection_idx])

        # Delete tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
