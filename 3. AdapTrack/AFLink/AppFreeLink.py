import time
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


INFINITY = 1e5


class AFLink:
    def __init__(self, path_in, path_out, model, dataset, thrT: tuple, thrS: int, thrP: float):
        self.thrP = thrP
        self.thrT = thrT
        self.thrS = thrS
        self.model = model
        self.dataset = dataset
        self.path_out = path_out
        self.track = np.loadtxt(path_in, delimiter=',')
        self.model.cuda()
        self.model.eval()

    def gather_info(self):
        id2info = defaultdict(list)
        self.track = self.track[np.argsort(self.track[:, 0])]

        for row in self.track:
            f, i, x, y, w, h = row[:6]
            id2info[i].append([f, x, y, w, h])

        self.track = np.array(self.track)
        id2info = {k: np.array(v) for k, v in id2info.items()}

        return id2info

    def compression(self, cost_matrix, ids):
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]

        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]

        return matrix, ids_row, ids_col

    def predict(self, track1, track2):
        track1, track2 = self.dataset.transform(track1, track2)
        track1, track2 = track1.unsqueeze(0).cuda(), track2.unsqueeze(0).cuda()
        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)
        return tracks[index]

    def link(self):
        id2info = self.gather_info()
        num = len(id2info)
        ids = np.array(list(id2info))
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)
        cost_matrix = np.ones((num, num)) * INFINITY

        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if id_i == id_j: continue

                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]

                if not self.thrT[0] <= fj - fi < self.thrT[1]:
                    continue

                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]):
                    continue

                cost = self.predict(info_i, info_j)

                if cost <= self.thrP:
                    cost_matrix[i, j] = cost

        id2id = dict()
        ID2ID = dict()
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)

        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]

        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k

        res = self.track.copy()
        for k, v in ID2ID.items():
            res[res[:, 1] == k, 1] = v

        res = self.deduplicate(res)

        start = time.time()
        np.savetxt(self.path_out, res, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        return time.time() - start
