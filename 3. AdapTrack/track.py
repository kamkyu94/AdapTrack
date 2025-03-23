import os
import time
import pickle
import warnings
import numpy as np
import random
import torch
from opts import Opts
from trackers.tracker import Tracker
from trackers.units import Detection
from trackers import metrics
from AFLink.AppFreeLink import AFLink
from interpolation.GSI import gsi_interpolation
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackeval.run import evaluate
from os.path import join

def create_detections(det_input, opt):
    detections = []
    if det_input is None:
        return detections
    for row in det_input:
        bbox = row[:4] if len(row) > 4 else row[0]
        confidence = row[4] if len(row) > 4 else row[1]
        feature = row[6:] if len(row) > 6 else (row[2] if len(row) > 2 else None)
        if confidence < opt.conf_thresh:
            continue
        detections.append(Detection(bbox, confidence, feature))
    return detections

def run(vid_name, save_path=None, opt=None, def_feat=None, detections_list=None):
    if opt is None:
        raise ValueError("opt must be provided to configure the Tracker.")
    metric = metrics.NearestNeighborDistanceMetric()
    tracker = Tracker(metric, vid_name, opt)
    results = []
    if detections_list is not None:
        frames = range(len(detections_list))
        detections_iter = [create_detections(frame_dets, opt) for frame_dets in detections_list]
    elif def_feat is not None:
        frames = sorted(def_feat.keys())
        detections_iter = [create_detections(def_feat[frame_idx], opt) for frame_idx in frames]
    else:
        raise ValueError("Either def_feat or detections_list must be provided.")
    for frame_idx, detections in zip(frames, detections_iter):
        frame_num = frame_idx + 1
        tracker.camera_update(frame_num)
        tracker.predict(frame_num)
        tracker.update(detections, frame_num)
        
        # Modified output logic
        frame_results = []
        if frame_idx == 0:  # First frame: include Tentative and Confirmed tracks
            output_tracks = [track for track in tracker.tracks if track.is_confirmed() or track.is_tentative()]
        else:  # Subsequent frames: include only Confirmed tracks
            output_tracks = [track for track in tracker.tracks if track.is_confirmed()]
        
        for track in output_tracks:
            if track.time_since_update > 1:  # Skip tracks not updated recently
                continue
            bbox = track.to_tlwh()
            if bbox[2] * bbox[3] > opt.min_box_area and bbox[2] / bbox[3] <= 1.6:  # Area and aspect ratio checks
                frame_results.append([frame_num, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        
        results.extend(frame_results)
        
        if frame_idx % 50 == 0:
            print(f'{vid_name} {frame_idx} / {len(frames)} Finished', flush=True)
    
    if save_path:
        start = time.time()
        with open(save_path, 'w') as f:
            for row in results:
                print(f'{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,-1,-1,-1', file=f)
        write_time = time.time() - start
    else:
        write_time = 0
    
    return write_time, len(frames)

def main(opt):
    af_linker = None
    if opt.AFLink:
        model = PostLinker()
        model.load_state_dict(torch.load(opt.AFLink_weight_path))
        dataset = LinkData('', '')
        af_linker = lambda path: AFLink(path_in=path, path_out=path, model=model, dataset=dataset,
                                        thrT=(0, 30), thrS=75, thrP=0.05)
    def_feat = {}
    if hasattr(opt, 'det_feat_path') and os.path.exists(opt.det_feat_path):
        with open(opt.det_feat_path, 'rb') as f:
            def_feat = pickle.load(f)
    total_time, total_img_num = 0, 0
    vid_names = opt.vid_names if hasattr(opt, 'vid_names') and opt.vid_names else ['default']
    for vid_name in vid_names:
        seq_info_path = join(opt.dataset_dir, vid_name, 'seqinfo.ini')
        if os.path.exists(seq_info_path):
            with open(seq_info_path, 'r') as seq_info:
                for s_i in seq_info.readlines():
                    if 'frameRate' in s_i:
                        opt.max_age = int(s_i.split('=')[-1]) * 2
                        break
        save_path = join(opt.save_dir, f'{vid_name}.txt')
        start = time.time()
        sub_time, img_num = run(vid_name=vid_name, def_feat=def_feat.get(vid_name), save_path=save_path, opt=opt)
        if opt.AFLink and af_linker:
            sub_time += af_linker(save_path).link()
        if opt.interpolation:
            sub_time += gsi_interpolation(save_path, save_path, interval=20, tau=10)
        total_time += ((time.time() - start) - sub_time)
        total_img_num += img_num
    time_per_img = total_time / total_img_num if total_img_num > 0 else 0
    print(f'Time per image: {time_per_img:.4f} sec, FPS: {1 / time_per_img:.2f}', flush=True)
    if opt.mode == 'val' and hasattr(opt, 'dataset'):
        setting_dict = {
            'gt_folder': join(opt.dataset_root, opt.dataset, 'train'),
            'gt_loc_format': '{gt_folder}/{seq}/gt/gt_val_half.txt',
            'trackers_folder': opt.save_dir.split('MOT')[0],
            'tracker': f'{opt.dataset}_{opt.mode}',
            'dataset': opt.dataset
        }
        evaluate(setting_dict)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(10000)
    random.seed(10000)
    opts = Opts()
    opt = opts.parse()
    main(opt)
    
    
    
    
    
    
    
    
    
    
    
    
    
