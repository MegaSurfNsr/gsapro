import numpy as np
import torch
import torch.utils.data as Dataset
import glob
import os
import tqdm
import cv2

def parse_cameras(path):
    cam_txt = open(path).readlines()
    f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5])
    intr_mat = f(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat

class cpuDB():
    def __init__(self,data_root,cam_root,pthres=0.35,tocuda=True,priorupscale=2,maskroot=None):
        self.hypo_list = glob.glob(os.path.join(data_root, 'hypo*'))
        self.hypo_list.sort()
        self.cost_list = glob.glob(os.path.join(data_root, 'cost*'))
        self.cost_list.sort()
        self.cams_list = glob.glob(os.path.join(cam_root, '*'))
        self.cams_list.sort()
        if maskroot is not None:
            self.masks_list = glob.glob(os.path.join(maskroot, '*.png'))
            self.masks_list.sort()
        self.depths = []
        for i in tqdm.tqdm(range(len(self.hypo_list))):
            helixout = np.load(self.hypo_list[i])
            cost = np.load(self.cost_list[i])
            extr_mat, intr_mat = parse_cameras(self.cams_list[i])
            intr_mat[:2,:] = intr_mat[:2,:] / priorupscale
            h,w,_ = cost.shape
            h_idx = np.arange(h, dtype=np.int32)
            w_idx = np.arange(w, dtype=np.int32)
            raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
            raw_grid.append(np.ones([h, w]))
            dep_map = -1 * helixout[:, :, -1] * intr_mat[0, 0] / (
                        (raw_grid[1] - intr_mat[0, 2]) * helixout[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
                            raw_grid[0] - intr_mat[1, 2]) * helixout[:, :, 1] + intr_mat[0, 0] * helixout[:, :, 2])
            dep_map[cost.squeeze(-1) >= pthres] = -1
            dep_map = cv2.resize(dep_map, dsize=None, fx=priorupscale, fy=priorupscale, interpolation=cv2.INTER_NEAREST)

            if maskroot is not None:
                mask = cv2.imread(self.masks_list[i],-1)
                if len(mask.shape) == 3:
                    mask = mask[:, :, -1] > 0
                dep_map[np.logical_not(mask)] = -1
            depth = torch.from_numpy(dep_map).unsqueeze(0)
            if tocuda:
                depth = depth.cuda()
            self.depths.append(depth)

    def __getitem__(self, index):
        return self.depths[index]