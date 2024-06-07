import torch
import numpy as np
from os.path import join
import AFLink.config as cfg
from torch.utils.data import Dataset
from random import randint, normalvariate


SEQ = {'train': ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',
                 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'],
       'test': ['MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN',
                'MOT17-08-FRCNN', 'MOT17-12-FRCNN', 'MOT17-14-FRCNN']}


class LinkData(Dataset):
    def __init__(self, root, mode='train', minLen=cfg.model_minLen, inputLen=cfg.model_inputLen):
        self.minLen = minLen
        self.inputLen = inputLen

        if root:
            assert mode in ('train', 'val')
            self.root = root
            self.mode = mode
            self.id2info = self.initialize()
            self.ids = list(self.id2info.keys())

    def initialize(self):
        id2info = dict()
        for seqid, seq in enumerate(SEQ['train'], start=1):
            path_gt = join(self.root, '{}/gt/gt_{}_half.txt'.format(seq, self.mode))
            gts = np.loadtxt(path_gt, delimiter=',')
            gts = gts[(gts[:, 6] == 1) * (gts[:, 7] == 1)]
            ids = set(gts[:, 1])

            for objid in ids:
                id_ = objid + seqid * 1e5
                track = gts[gts[:, 1] == objid]
                fxywh = [[t[0], t[2], t[3], t[4], t[5]] for t in track]

                if len(fxywh) >= self.minLen:
                    id2info[id_] = np.array(fxywh)

        return id2info

    def fill_or_cut(self, x, former: bool):
        lengthX, widthX = x.shape

        if lengthX >= self.inputLen:
            if former:
                x = x[-self.inputLen:]
            else:
                x = x[:self.inputLen]
        else:
            zeros = np.zeros((self.inputLen - lengthX, widthX))
            if former:
                x = np.concatenate((zeros, x), axis=0)
            else:
                x = np.concatenate((x, zeros), axis=0)

        return x

    def transform(self, x1, x2):
        # fill or cut
        x1 = self.fill_or_cut(x1, True)
        x2 = self.fill_or_cut(x2, False)

        # min-max normalization
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor

        # numpy to torch
        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)

        # unsqueeze channel=1
        x1 = x1.unsqueeze(dim=0)
        x2 = x2.unsqueeze(dim=0)
        return x1, x2

    def __getitem__(self, item):
        info = self.id2info[self.ids[item]]
        numFrames = info.shape[0]

        if self.mode == 'train':
            idxCut = randint(self.minLen//3, numFrames - self.minLen//3)

            info1 = info[:idxCut + int(normalvariate(-5, 3))]
            info2 = info[idxCut + int(normalvariate(5, 3)):]

            info2_t = info2.copy()
            info2_t[:, 0] += (-1) ** randint(1, 2) * randint(30, 100)

            info2_s = info2.copy()
            info2_s[:, 1] += (-1) ** randint(1, 2) * randint(100, 500)
            info2_s[:, 2] += (-1) ** randint(1, 2) * randint(100, 500)

        else:
            idxCut = numFrames // 2

            info1 = info[:idxCut]
            info2 = info[idxCut:]

            info2_t = info2.copy()
            info2_t[:, 0] += (-1) ** idxCut * 50

            info2_s = info2.copy()
            info2_s[:, 1] += (-1) ** idxCut * 300
            info2_s[:, 2] += (-1) ** idxCut * 300

        return self.transform(info1, info2), self.transform(info2, info1), self.transform(info1, info2_t), \
               self.transform(info1, info2_s), (1, 0, 0, 0)

    def __len__(self):
        return len(self.ids)
