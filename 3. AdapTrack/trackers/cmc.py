import numpy as np

class CMC:
    def __init__(self, vid_name):
        super(CMC, self).__init__()
        self.gmcFile = open('./trackers/cmc/GMC-' + vid_name + '.txt', 'r')

    def get_warp_matrix(self):
        line = self.gmcFile.readline()
        if not line:
            return np.eye(2, 3, dtype=np.float_)
        tokens = line.strip().split("\\t")
        if len(tokens) < 7:
            print(f"Warning: Malformed line in GMC file: {line.strip()}")
            return np.eye(2, 3, dtype=np.float_)
        warp_matrix = np.eye(2, 3, dtype=np.float_)
        warp_matrix[0, 0] = float(tokens[1])
        warp_matrix[0, 1] = float(tokens[2])
        warp_matrix[0, 2] = float(tokens[3])
        warp_matrix[1, 0] = float(tokens[4])
        warp_matrix[1, 1] = float(tokens[5])
        warp_matrix[1, 2] = float(tokens[6])
        return warp_matrix