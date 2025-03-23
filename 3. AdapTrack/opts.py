import os
import argparse
from os.path import join

class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="AdapTrack Configuration")
        self.parser.add_argument('--dataset', type=str, default='MOT17', help='Dataset name (e.g., MOT17, MOT20, custom)')
        self.parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'], help='Mode: val or test')
        self.parser.add_argument('--dataset_root', type=str, default='../../dataset/', help='Root directory of dataset')
        self.parser.add_argument('--save_dir', type=str, default='./outputs/', help='Output directory')
        self.parser.add_argument('--max_distance', type=float, default=0.45, help='Max distance for tracking')
        self.parser.add_argument('--max_iou_distance', type=float, default=0.70, help='Max IoU distance for tracking')
        self.parser.add_argument('--min_len', type=int, default=3, help='Min track length to confirm')
        self.parser.add_argument('--max_age', type=int, default=30, help='Max age of unupdated tracks')
        self.parser.add_argument('--max_age_tentative', type=int, default=2, help='Max age for tentative tracks before deletion')
        self.parser.add_argument('--ema_beta', type=float, default=0.90, help='EMA smoothing factor')
        self.parser.add_argument('--gating_lambda', type=float, default=0.98, help='Gating parameter')
        self.parser.add_argument('--min_box_area', type=int, default=100, help='Min box area for valid tracks')
        self.parser.add_argument('--AFLink', action='store_true', default=True, help='Enable AFLink post-processing')
        self.parser.add_argument('--interpolation', action='store_true', default=True, help='Enable interpolation')

    def parse(self, args=None):
        # Allow passing custom args or use sys.argv
        if args is None:
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        
        # Set derived attributes
        opt.save_dir += '%s_%s' % (opt.dataset, opt.mode)
        opt.dataset_dir = join(opt.dataset_root, opt.dataset, 'train' if opt.mode == 'val' else 'test')
        opt.det_feat_path = '../outputs/2. det_feat/%s_%s.pickle' % (opt.dataset, opt.mode)
        opt.AFLink_weight_path = './AFLink/AFLink_epoch20.pth'
        opt.conf_thresh = 0.6 if opt.dataset == 'MOT17' else 0.4
        
        # Only set vid_names if dataset_dir exists, otherwise leave it unset
        if os.path.exists(opt.dataset_dir):
            opt.vid_names = os.listdir(opt.dataset_dir)
        else:
            opt.vid_names = None
        
        os.makedirs(opt.save_dir, exist_ok=True)
        return opt