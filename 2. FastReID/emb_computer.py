import cv2
import torch
import numpy as np
from fastreid.fastreid_adaptor import FastReID


class EmbeddingComputer:
    def __init__(self, dataset, max_batch=1024):
        self.model = None
        self.dataset = dataset
        self.crop_size = (128, 384)
        self.max_batch = max_batch

    def initialize_model(self):
        # Set path
        path = "weights/%s_sbs_S50.pth" % self.dataset

        # Set model
        print('Pre-trained weight: %s' % path)
        self.model = FastReID(self.dataset, path)

    def compute_embedding(self, img, bbox):
        # Initialization
        if self.model is None:
            self.initialize_model()

        # Basic embeddings
        h, w = img.shape[:2]
        bbox_clip = np.round(bbox).astype(np.int32)
        bbox_clip[:, 0] = bbox_clip[:, 0].clip(0, w)
        bbox_clip[:, 1] = bbox_clip[:, 1].clip(0, h)
        bbox_clip[:, 2] = bbox_clip[:, 2].clip(0, w)
        bbox_clip[:, 3] = bbox_clip[:, 3].clip(0, h)

        # Get patches
        crops = []
        for box in bbox_clip:
            # Get patch, BGR -> RGB, Resize
            crop = img[box[1]:box[3], box[0]:box[2]]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

            # To Tensor, Append
            crop = torch.as_tensor(crop.transpose(2, 0, 1)).unsqueeze(0).cuda()
            crops.append(crop)

        # To batch
        crops = torch.cat(crops, dim=0)

        # Get embeddings
        embeddings = []
        for idx in range(0, len(crops), self.max_batch):
            # Get batch
            batch_crops = crops[idx:idx + self.max_batch]
            batch_crops = batch_crops.cuda()

            # Inference
            with torch.no_grad():
                batch_embeddings = self.model(batch_crops)

            # Add (extend)
            embeddings.extend(batch_embeddings)

        # Stack, L2 Normalize, To numpy
        embeddings = torch.stack(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.cpu().numpy()

        return embeddings
