import numpy as np
from trackers.cmc import CMC
from trackers import metrics
from trackers.units import Track
from trackers import linear_assignment

def apply_cmc(track, warp_matrix):
    pass

class Tracker:
    def __init__(self, metric, vid_name, opt):
        self.metric = metric
        self.opt = opt
        self.tracks = []
        self.next_id = 1
        try:
            self.cmc = CMC(vid_name)
            print(f"CMC initialized for {vid_name}")
        except FileNotFoundError:
            print(f"No GMC file found for {vid_name}, disabling camera motion compensation.")
            self.cmc = None

    def initiate_track(self, detection, frame_idx):
        self.tracks.append(Track(detection.to_cxcyah(), self.next_id, self.opt, detection.confidence, detection.feature))
        #print(f"Frame {frame_idx}: Initiated track {self.next_id} with confidence {detection.confidence}")
        self.next_id += 1

    def predict(self, frame_idx):
        for track in self.tracks:
            track.predict()
        print(f"Frame {frame_idx}: Predicted {len(self.tracks)} tracks")

    def camera_update(self, frame_idx):
        if self.cmc is not None:
            warp_matrix = self.cmc.get_warp_matrix()
            print(f"Frame {frame_idx}: Applying CMC with warp matrix {warp_matrix}")
            for track in self.tracks:
                apply_cmc(track, warp_matrix)
        else:
            print(f"Frame {frame_idx}: No CMC applied")

    def gated_metric(self, tracks, detections, track_indices, detection_indices, frame_idx):
        targets = np.array([tracks[i].track_id for i in track_indices])
        features = np.array([detections[i].feature for i in detection_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix_min = np.min(cost_matrix)
        cost_matrix_max = np.max(cost_matrix)
        cost_matrix = linear_assignment.gate_cost_matrix(
            cost_matrix, tracks, detections, track_indices, detection_indices, self.opt.gating_lambda
        )
        print(f"Frame {frame_idx}: Gated metric - Min cost: {cost_matrix_min}, Max cost: {cost_matrix_max}, Shape: {cost_matrix.shape}")
        return cost_matrix, cost_matrix_min, cost_matrix_max

    def match(self, detections, frame_idx):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        print(f"Frame {frame_idx}: Confirmed tracks: {len(confirmed_tracks)}, Unconfirmed tracks: {len(unconfirmed_tracks)}")
        
        matches_a, _, unmatched_detections = linear_assignment.min_cost_matching(
            [lambda t, d, ti, di: self.gated_metric(t, d, ti, di, frame_idx), metrics.iou_constraint, True],
            self.opt.max_distance,
            self.tracks,
            detections,
            confirmed_tracks
        )
        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _ in matches_a))
        candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            [metrics.iou_cost, None, True],
            self.opt.max_iou_distance,
            self.tracks,
            detections,
            candidates,
            unmatched_detections
        )
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        print(f"Frame {frame_idx}: Matches: {len(matches)}, Unmatched tracks: {len(unmatched_tracks)}, Unmatched detections: {len(unmatched_detections)}")
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections, frame_idx):
        print(f"Frame {frame_idx}: Received {len(detections)} detections")
        matches, unmatched_tracks, unmatched_detections = self.match(detections, frame_idx)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
            #print(f"Frame {frame_idx}: Updated track {self.tracks[track_idx].track_id}, confirmed: {self.tracks[track_idx].is_confirmed()}")
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            print(f"Frame {frame_idx}: Marked track {self.tracks[track_idx].track_id} as missed, confirmed: {self.tracks[track_idx].is_confirmed()}")
        for detection_idx in unmatched_detections:
            if detections[detection_idx].confidence >= self.opt.conf_thresh:
                self.initiate_track(detections[detection_idx], frame_idx)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        print(f"Frame {frame_idx}: Active confirmed tracks: {len(active_targets)}")
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)