import torch
import numpy as np
from tqdm import tqdm
from yolox.utils import (is_main_process, postprocess,)


class DetEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (tuple): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def detect(self, model, half=False):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()

        det_results = {}
        progress_bar = tqdm if is_main_process() else iter
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                video_name = info_imgs[4][0].split('/')[2]
                frame_id = info_imgs[2].item()
                if video_name not in det_results.keys():
                    det_results[video_name] = {}

                imgs = imgs.type(tensor_type)
                outputs = model(imgs)
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

            # outputs[0]: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            det = outputs[0]
            if det is not None:
                # if 600 < len(det):
                #     confidence = det[:, 4] * det[:, 5]
                #     det = det[torch.topk(confidence, 600, dim=0)[1], :]

                det[:, 4] *= det[:, 5]
                det[:, 5] = det[:, 6]
                det = det[:, :6]

                img_h, img_w = info_imgs[0], info_imgs[1]
                img_h, img_w = float(img_h), float(img_w)
                scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))

                det = det.detach().cpu().numpy()

                det[:, :4] /= scale

                # Filter
                det = det[(np.minimum(det[:, 2], img_w - 1) - np.maximum(det[:, 0], 0)) > 0]
                det = det[(np.minimum(det[:, 3], img_h - 1) - np.maximum(det[:, 1], 0)) > 0]
                # det = det[((np.minimum(det[:, 2], img_w - 1) - np.maximum(det[:, 0], 0))
                #            * (np.minimum(det[:, 3], img_h - 1) - np.maximum(det[:, 1], 0))) > self.args.min_box_area]

                det_results[video_name][frame_id] = det if len(det) > 0 else None
            else:
                det_results[video_name][frame_id] = None

        return det_results
