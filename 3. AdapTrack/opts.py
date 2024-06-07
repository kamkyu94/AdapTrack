import os
import argparse
from os.path import join


class Opts:
    def __init__(self):
        # Initialize
        self.parser = argparse.ArgumentParser()

        # Basic Setting
        self.parser.add_argument('--dataset', default='MOT17', help='MOT17, MOT20')
        self.parser.add_argument('--mode', default='val', help='val or test')
        self.parser.add_argument('--dataset_root', default='../../dataset/')
        self.parser.add_argument('--save_dir', default='./outputs/')

        # For tracking
        self.parser.add_argument('--max_distance', default=0.45)
        self.parser.add_argument('--max_iou_distance', default=0.70)
        self.parser.add_argument('--min_len', default=3)
        self.parser.add_argument('--max_age', default=30)
        self.parser.add_argument('--ema_beta', default=0.90)
        self.parser.add_argument('--gating_lambda', default=0.98)
        self.parser.add_argument("--min_box_area", default=100)

        # For Post-processing
        self.parser.add_argument('--AFLink', default=True, action='store_true')
        self.parser.add_argument('--interpolation', default=True, action='store_true')

    def parse(self):
        # Initialize
        opt = self.parser.parse_args()

        # Set directories, paths
        opt.save_dir += '%s_%s' % (opt.dataset, opt.mode)
        opt.dataset_dir = join(opt.dataset_root, opt.dataset, 'train' if opt.mode == 'val' else 'test')
        opt.det_feat_path = '../outputs/2. det_feat/%s_%s.pickle' % (opt.dataset, opt.mode)
        opt.AFLink_weight_path = './AFLink/AFLink_epoch20.pth'

        # Set others
        opt.conf_thresh = 0.6 if opt.dataset == 'MOT17' else 0.4
        opt.vid_names = os.listdir(opt.dataset_dir)

        # Make dir
        os.makedirs(opt.save_dir, exist_ok=True)

        return opt


# Create option file
opt = Opts().parse()
