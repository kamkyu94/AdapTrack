import numpy as np


class CMC:
    def __init__(self, vid_name):
        super(CMC, self).__init__()
        self.gmcFile = open('./trackers/cmc/GMC-' + vid_name + '.txt', 'r')

    def get_warp_matrix(self):
        # Read line and separate
        line = self.gmcFile.readline()
        tokens = line.split("\t")

        # To warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float_)
        warp_matrix[0, 0] = float(tokens[1])
        warp_matrix[0, 1] = float(tokens[2])
        warp_matrix[0, 2] = float(tokens[3])
        warp_matrix[1, 0] = float(tokens[4])
        warp_matrix[1, 1] = float(tokens[5])
        warp_matrix[1, 2] = float(tokens[6])

        return warp_matrix


def apply_cmc(track, warp_matrix=np.eye(2, 3)):
    # Get box
    x1, y1, x2, y2 = track.to_tlbr()

    # Warp
    x1_, y1_ = warp_matrix @ np.array([x1, y1, 1]).T
    x2_, y2_ = warp_matrix @ np.array([x2, y2, 1]).T

    # Re-calculate cx, cy, w, h
    w, h = x2_ - x1_, y2_ - y1_
    cx, cy = x1_ + w / 2, y1_ + h / 2

    # Change mean (cx, cy, a, h)
    track.mean[:4] = [cx, cy, w / h, h]
