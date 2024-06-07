import torch
import random
import pickle
import warnings
from opts import *
from os.path import join
from trackers import metrics
from AFLink.AppFreeLink import *
from trackeval.run import evaluate
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker import Tracker
from trackers.units import Detection
from interpolation.GSI import gsi_interpolation


def create_detections(det_feat):
    # Initialization
    detections = []

    # Check
    if det_feat is None:
        return detections

    # Generate detections
    for row in det_feat:
        # Get Information
        bbox, confidence, feature = row[:4], row[4], row[6:]

        # Filter detection with confidence score (Without Byte)
        if confidence < opt.conf_thresh:
            continue

        # Append
        detections.append(Detection(bbox, confidence, feature))

    return detections


def run(vid_name, def_feat, save_path):
    # Set
    metric = metrics.NearestNeighborDistanceMetric()
    tracker = Tracker(metric, vid_name)
    results = []

    # Run
    for frame_idx in def_feat.keys():
        # Generate detections
        detections = create_detections(def_feat[frame_idx])

        # Camera motion compensation
        tracker.camera_update()

        # Update trackers
        tracker.predict()
        tracker.update(detections)

        # Store proper results
        for track in tracker.tracks:
            # Check
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Check
            bbox = track.to_tlwh()
            if bbox[2] * bbox[3] > opt.min_box_area and bbox[2] / bbox[3] <= 1.6:
                results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        # Logging
        if frame_idx % 50 == 0:
            print('%s %d / %d Finished' % (vid_name, frame_idx, len(def_feat.keys())), flush=True)

    # Write results
    start = time.time()
    f = open(save_path, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
    f.close()

    return time.time() - start, len(def_feat.keys())


def main():
    # Initialize AFLink
    if opt.AFLink:
        model = PostLinker()
        model.load_state_dict(torch.load(opt.AFLink_weight_path))
        dataset = LinkData('', '')

    # Read detection and feature
    with open(opt.det_feat_path, 'rb') as f:
        def_feat = pickle.load(f)

    # Initialization
    total_time, total_img_num = 0, 0

    # Run
    for sdx, vid_name in enumerate(opt.vid_names, start=1):
        # Set max_age
        seq_info = open(opt.dataset_dir + '/' + vid_name + '/seqinfo.ini', mode='r')
        for s_i in seq_info.readlines():
            if 'frameRate' in s_i:
                opt.max_age = int(s_i.split('=')[-1]) * 2
                break

        # Set path to save
        save_path = join(opt.save_dir, vid_name + '.txt')

        # Measure time
        start = time.time()

        # Track
        sub_time, img_num = run(vid_name=vid_name,
                                def_feat=def_feat[vid_name],
                                save_path=save_path)

        # Post-processing
        if opt.AFLink:
            linker = AFLink(path_in=save_path, path_out=save_path, model=model, dataset=dataset,
                            thrT=(0, 30), thrS=75, thrP=0.05)
            sub_time += linker.link()

        # Post-processing
        if opt.interpolation:
            sub_time += gsi_interpolation(save_path, save_path, interval=20, tau=10)

        # Update
        total_time += ((time.time() - start) - sub_time)
        total_img_num += img_num

    # Logging
    time_per_img = total_time / total_img_num
    print('Time per image: %.4f sec, FPS: %f' % (time_per_img, 1 / time_per_img), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.random.seed(10000)
    random.seed(10000)

    # Run
    main()

    if opt.mode == 'val':
        # Evaluation
        setting_dict = {'gt_folder': opt.dataset_root + opt.dataset + '/train',
                        'gt_loc_format': '{gt_folder}/{seq}/gt/gt_val_half.txt',
                        'trackers_folder': opt.save_dir.split('MOT')[0],
                        'tracker': opt.dataset + '_' + opt.mode,
                        'dataset': opt.dataset}
        # Evaluate
        evaluate(setting_dict)
