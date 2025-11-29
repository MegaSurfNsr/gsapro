import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(parent_dir)

parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_parent_dir)

import cv2
import torch
import matplotlib.pyplot as plt
from helix_model.helixncc import HelixNCC,transform_normalmap
import glob

from datetime import datetime
import random
import tqdm
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=str, default=None)
args = parser.parse_args()

if __name__ == "__main__":
    dataroot = '/mnt/data4/yswangdata4/dataset/TankandTemples/intermediate'
    caselist = '/mnt/data4/yswangdata4/code/mvsformerpp_original/lists/tanksandtemples/intermediate.txt'
    saveroot = '/mnt/data4/yswangdata4/dataset/TankandTemples/pm_intermediate'

    if args.case is None:
        with open(caselist,'r') as f:
            validscenes = f.readlines()
        validscenes = [c.strip() for c in validscenes]
    else:
        validscenes = [args.case.strip()]

    for scene in tqdm.tqdm(validscenes):
        sceneroot = os.path.join(dataroot,scene)
        imgpath = f'{sceneroot}/images'
        camspath = f'{dataroot}/short_range_cameras/cams_{scene.lower()}'
        pairpath = os.path.join(sceneroot,'new_pair.txt')
        savepath = os.path.join(f'{saveroot}/{scene}', 'helixout')
        os.makedirs(savepath,exist_ok=True)

        ncc_pthred = 0.25
        helixncc = HelixNCC(camspath,pairpath,None,imgpath,dscale=1,pairneighbor=20)
        dist_thresh = np.asarray([c[0][3,3] - c[0][3,0] for c in helixncc.cams]).mean() * 0.01
        depth_filter_param = {
            'dist_thresh': dist_thresh,  # 0.05
            'scale': 1,
            'prob_thresh': 0.25,
            'num_consist': 7,
            'device': 'cuda',
            'srcs': 'all',  # ['pair','all']
        }

        helixncc.depth_filter_param = depth_filter_param

        for i in tqdm.tqdm(range(helixncc.n_image)):
            helixncc.process_pair_data(i)
            helixout = helixncc.hypos.cpu().numpy()
            cost = helixncc.cost.cpu().numpy()
            helixncc.priors.append(helixout)

            intr_mat = helixncc.cams[i][0][:3,:3]
            h, w, _ = cost.shape
            h_idx = np.arange(h, dtype=np.int32)
            w_idx = np.arange(w, dtype=np.int32)
            raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
            raw_grid.append(np.ones([h, w]))
            dep_map = -1 * helixout[:, :, -1] * intr_mat[0, 0] / (
                    (raw_grid[1] - intr_mat[0, 2]) * helixout[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
                    raw_grid[0] - intr_mat[1, 2]) * helixout[:, :, 1] + intr_mat[0, 0] * helixout[:, :, 2])
            pthres = ncc_pthred

            depth = torch.from_numpy(dep_map).unsqueeze(0)
            cost = torch.from_numpy(cost).unsqueeze(0).squeeze(-1)

            helixncc.depths.append(depth)
            helixncc.costs.append(cost)

            np.save(os.path.join(savepath,f'depth_{i.__str__().zfill(8)}_i3.npy'),np.asarray(helixncc.depths[i],dtype=np.float32))
            np.save(os.path.join(savepath,f'cost_{i.__str__().zfill(8)}_i3.npy'),np.asarray(helixncc.costs[i],dtype=np.float32))

        # save


        helixncc.depth_reproj_filter()
        for i in range(helixncc.n_image):
            fmask = np.asarray(helixncc.filter_depth_mask[i]*255,dtype=np.uint8)[0]
            cv2.imwrite(os.path.join(savepath,f'mask_{i.__str__().zfill(8)}_i3.png'),fmask)
        print(f'process done {scene}')


        # helixncc.depths = []
        # helixncc.costs = []
        # for i in tqdm.tqdm(range(helixncc.n_image)):
        #     helixncc.process_with_prior(i)
        #     helixout = helixncc.hypos.cpu().numpy()
        #     cost = helixncc.cost.cpu().numpy()
        #     helixncc.priors.append(helixout)
        #
        #     intr_mat = helixncc.cams[i][0][:3,:3]
        #     h, w, _ = cost.shape
        #     h_idx = np.arange(h, dtype=np.int32)
        #     w_idx = np.arange(w, dtype=np.int32)
        #     raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
        #     raw_grid.append(np.ones([h, w]))
        #     dep_map = -1 * helixout[:, :, -1] * intr_mat[0, 0] / (
        #             (raw_grid[1] - intr_mat[0, 2]) * helixout[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
        #             raw_grid[0] - intr_mat[1, 2]) * helixout[:, :, 1] + intr_mat[0, 0] * helixout[:, :, 2])
        #     pthres = ncc_pthred
        #
        #     depth = torch.from_numpy(dep_map).unsqueeze(0)
        #     cost = torch.from_numpy(cost).unsqueeze(0).squeeze(-1)
        #
        #     helixncc.depths.append(depth)
        #     helixncc.costs.append(cost)
        #
        #     np.save(os.path.join(savepath,f'depth_{i.__str__().zfill(8)}.npy'),np.asarray(helixncc.depths[i],dtype=np.float32))
        #     np.save(os.path.join(savepath,f'cost_{i.__str__().zfill(8)}.npy'),np.asarray(helixncc.costs[i],dtype=np.float32))
        #
        # helixncc.depth_reproj_filter()
        # for i in range(helixncc.n_image):
        #     fmask = np.asarray(helixncc.filter_depth_mask[i]*255,dtype=np.uint8)[0]
        #     cv2.imwrite(os.path.join(savepath,f'mask_{i.__str__().zfill(8)}.png'),fmask)


