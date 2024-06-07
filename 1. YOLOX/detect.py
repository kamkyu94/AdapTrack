import os
import torch
import random
import pickle
import argparse
import numpy as np
from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import fuse_model
import torch.backends.cudnn as cudnn
from yolox.evaluators import DetEvaluator
from torch.nn.parallel import DistributedDataParallel as DDP


def make_parser():
    parser = argparse.ArgumentParser("YOLOX")

    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your experiment description file",)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn",)
    parser.add_argument("-t", "--type", default=None, type=str)
    parser.add_argument("-n", "--exp_name", type=str, default=None)

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training",)
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER,)
    parser.add_argument("--fp16", dest="fp16", action="store_true",)

    # det args
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.8, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--min_box_area", default=100, type=int, help="filter out tiny boxes")
    parser.add_argument("--seed", default=10000, type=int, help="eval seed")

    return parser


def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Added
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        ckpt_file = args.ckpt
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
    if is_distributed:
        model = DDP(model, device_ids=[rank])
    if args.fuse:
        model = fuse_model(model)

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = DetEvaluator(args=args, dataloader=val_loader, img_size=exp.test_size, confthre=exp.test_conf,
                             nmsthre=exp.nmsthre, num_classes=exp.num_classes,)

    # start evaluate, x1y1x2y2
    det_results = evaluator.detect(model, args.fp16)

    with open(args.exp_name, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)
    exp.merge(args.opts)

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
