import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from ysutils.util_acmh import read_dmb
from ysutils.util_mvsnet import load_cam
from ysutils.util_colmap import qvec2rotmat
import ncc
import argparse
import tqdm
import copy
import sys
from scipy.spatial.transform import Rotation, RotationSpline
from depthfusion.warp_func import *


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''

    :param ref_depth: (1, 1, H, W)
    :param src_depths: (B, 1, H, W)
    :param ref_proj: (1, 4, 4)
    :param src_proj: (B, 4, 4)
    :return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
    '''

    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    # _, axs = plt.subplots(3, 4)
    # for i in range(3):
    # 	axs[i, 0].imshow(src_pcs[0, i], vmin=0, vmax=1)
    # 	axs[i, 1].imshow(aligned_pcs[0, i],  vmin=0, vmax=1)
    # 	axs[i, 2].imshow(ref_pc[0, i],  vmin=0, vmax=1)
    # 	axs[i, 3].imshow(ref_pc[0, i] - aligned_pcs[0, i], vmin=-0.5, vmax=0.5, cmap='coolwarm')
    # plt.show()

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist

def generateRandNoise(hypo,K):
    h, w, _ = hypo.shape
    h_idx = np.arange(h, dtype=np.int32)
    w_idx = np.arange(w, dtype=np.int32)
    raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
    raw_grid.append(np.ones([h, w]))
    x = (raw_grid[1] - K[0, 2]) / K[0, 0]
    y = (raw_grid[0] - K[1, 2]) / K[1, 1]
    z = np.ones([h, w])
    perturb_rot = Rotation.random(h*w)

    rot = Rotation.from_rotvec(hypo[:,:,:3].reshape(-1,3))
    rot.as_euler('xyz', degrees=True)


def read_img_dtu(root,dscale=1):
    imlist = []
    flist = glob.glob(os.path.join(root, '*.jpg'))
    if len(flist) == 0:
        flist = glob.glob(os.path.join(root, '*.png'))
    assert len(flist) != 0, 'image root err!'
    flist.sort()

    filter_flist = []
    for f in flist:
        if f.split('_')[2] == '3':
            filter_flist.append(f)
    filter_flist.sort()
    for f_im in filter_flist:
        temimg = cv2.imread(f_im,cv2.IMREAD_GRAYSCALE)
        temimg = cv2.resize(temimg,dsize=None,fx=1/dscale,fy=1/dscale)
        imlist.append(temimg)
    return np.ascontiguousarray(np.asarray(imlist))


def read_img(root,dscale=1,parallel=False):
    imlist = []
    flist = glob.glob(os.path.join(root, '*.jpg'))
    if len(flist) == 0:
        flist = glob.glob(os.path.join(root, '*.png'))

    if len(flist) == 0:
        flist = glob.glob(os.path.join(root,'*.JPG'))

    assert len(flist) != 0, 'image root err!'
    flist.sort()
    for f_im in flist:
        temimg = cv2.imread(f_im,cv2.IMREAD_GRAYSCALE)
        temimg = cv2.resize(temimg,dsize=None,fx=1/dscale,fy=1/dscale)
        imlist.append(temimg)
    return np.ascontiguousarray(np.asarray(imlist))

def load_all_cams(root,dscale=1,changeTK=True):
    '''

    :param root:
    :param changeTK: change the idx of T and K. (From [n,2(T,K),4,4] to [n,2(K,T),4,4] )
    :return: np [n,2,4,4]
    '''
    cams = []
    flist = glob.glob(os.path.join(root, '*cam.txt'))
    flist.sort()
    for f_cam in flist:
        temcam = np.zeros((2, 4, 4))
        cam = load_cam(f_cam)
        cam[1][:2,:] = cam[1][:2,:]/dscale
        if changeTK:
            temcam[0] = cam[1]
            temcam[1] = cam[0]
        cams.append(temcam)
    return np.ascontiguousarray(np.asarray(cams,dtype=np.float32))

def parse_pair(file,pairneighbor=10,format='acmh'):
    if format == 'acmh':
        with open(file,'r') as f:
            lines = f.readlines()
        pairs = {}
        n = int(lines[0].strip())
        for i in range(1,n*2 +1,2):
            id = int(lines[i].strip())
            pair = lines[i+1].strip().split(' ')
            n_pair = int(pair[0].strip())
            valid_n_pair = (len(pair) - 1)//2
            if valid_n_pair != pairneighbor:
                print(f'warning: only find {valid_n_pair} pairs for id {id}')
            pairid = []
            additional_id = 1
            for j in range(1,n_pair*2+1,2):
                if j < valid_n_pair*2+1:
                    pairid.append(int(pair[j]))
                else:
                    pairid.append(int(pair[additional_id]))
                    additional_id = additional_id + 2

            pairs[id] = pairid
    else:
        pass
    return pairs

def load_pgsr_output(path):
    files = glob.glob(os.path.join(path,'*.npy'))
    files.sort()
    priors = [np.load(f) for f in files]
    return priors

def transform_normalmap(normal_map,pose):
    norm_acmh_rot = pose[:3,:3]
    return np.matmul(norm_acmh_rot[None, None, :, :],normal_map[:, :, :, None]).squeeze(-1)

def generate_plane_hypo(depthnormal,K):
    h,w,_ = depthnormal.shape
    h_idx = np.arange(h, dtype=np.int32)
    w_idx = np.arange(w, dtype=np.int32)
    raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
    raw_grid.append(np.ones([h, w]))
    x = depthnormal[:,:,0] * (raw_grid[1] - K[0,2]) / K[0,0]
    y = depthnormal[:, :, 0] * (raw_grid[0] - K[1, 2]) / K[1, 1]
    z = depthnormal[:,:,0]
    w = -1 * (depthnormal[:, :, 1] * x + depthnormal[:, :, 2] * y + depthnormal[:, :, 3] * z)
    return np.concatenate([depthnormal[:,:,1:],w[:,:,None]], axis=-1)


class HelixNCC():
    def __init__(self,cams_path,pairs_path,priors_path,images_path,dscale=1,pairneighbor=10,read_dtu=False):
        self.pairs_dict = parse_pair(pairs_path,pairneighbor)
        self.cams = load_all_cams(cams_path,dscale=dscale)
        self.priors = []
        self.depths = []
        self.costs = []
        self.seg_mask = []
        self.masks = []
        self.helix_ratio = 0.1
        self.meta = {}
        self.fake_pair_path = "/mnt/data3/yswang2024_data3/code/helixgau/fake_pair.txt"
        self.maskflag = False
        self.depth_filter_param = {
            'dist_thresh': 0.05, # 0.05
            'scale': 1,
            'prob_thresh': 0.4,
            'num_consist': 3,
            'device': 'cuda',
            'srcs':'pair', # ['pair','all']
        }
        if pairneighbor == 20:
            self.fake_pair_path = "/mnt/data3/yswang2024_data3/code/helixgau/fake_pair_20.txt"
        if priors_path is not None:
            self.priors = load_pgsr_output(priors_path)
            for i in range(len(self.priors)):
                self.priors[i] = generate_plane_hypo(self.priors[i],self.cams[i][0])

        # change pgsr(camera) normal to acmh normal (global)
        # poses = [np.linalg.pinv(self.cams[i][1]) for i in range(len(self.cams))]
        # for i in tqdm.tqdm(range(len(self.priors))):
        #     self.priors[i][:, :, 1:] = transform_normalmap(self.priors[i][:, :, 1:], poses[i])
            # plt.imshow(priors[20][:, :, 1:] * 0.5 + 0.5)
            # plt.show()

        # dacmh = read_dmb('/mnt/data2/yswang2024/dataset/scan24_2dgs/ACMH/2333_00000032/depths.dmb')
        # plt.imshow(dacmh[180:400,1100:1300,0].clip(1.2,2.8))
        # plt.show()
        # plt.imshow(priors[32][180:400,1100:1300,0].clip(1.2,2.8))
        # plt.show()
        #
        if read_dtu:
            self.ims = read_img_dtu(images_path, dscale=dscale)
        else:
            self.ims = read_img(images_path, dscale=dscale)
        self.ims = np.asarray(self.ims / 255., dtype=np.float32)  # 不做归一化可以大幅减少nan
        # print(self.cams.flags.c_contiguous)
        # print(self.ims.flags.c_contiguous)

        self.n_image, self.h, self.w = self.ims.shape
        n = len(self.pairs_dict[0]) + 1
        self.py_depths = np.zeros((n, self.h, self.w), dtype=np.float32)
        self.py_vselects = np.zeros((n, self.h, self.w), dtype=np.uint32)
        self.py_costs = np.ones((n, self.h, self.w), dtype=np.float32)
        self.py_planeHypos = np.zeros((n, self.h, self.w, 4), dtype=np.float32)
        self.n_buffer = n

        # h_idx = np.arange(self.h, dtype=np.int32)
        # w_idx = np.arange(self.w, dtype=np.int32)
        # raw_grid = np.meshgrid(h_idx, w_idx, indexing="ij")
        # self.coord_tem = np.ascontiguousarray(np.stack([raw_grid[0], raw_grid[1]], axis=-1).reshape(-1, 2).copy())  # h,w
        # print(self.coord_tem.dtype)
        # self.coord_num_tem = self.coord_tem.shape[0]

        self.hypos = torch.zeros([self.h,self.w,4], device='cuda')
        self.cost = torch.zeros([self.h,self.w,1], device='cuda')

    def process_pair_data(self,idx):
        # usage
        # initialization
        ncc_module = ncc.pyNCCmodule()
        # ncc_module.__version__()
        ncc_module.set_hw(self.h, self.w)
        ncc_module.set_set_pert_depth_ratio(self.helix_ratio)
        # load the pair. info
        ncc_module.load_pair(self.fake_pair_path)
        ids = [idx] + self.pairs_dict[idx]
        cams = np.ascontiguousarray(self.cams[ids])
        ims = np.ascontiguousarray(self.ims[ids])

        if len(ids) < self.n_buffer:
            print(f'warning! src imgs for {idx} not enough!')
            n_now = len(ids)
            add_cams = []
            add_ims = []
            add_ids = []
            ptr = 1
            while n_now < self.n_buffer:
                add_idx = ids[ptr]
                add_ids.append(add_idx)
                add_cams.append(self.cams[add_idx])
                add_ims.append(self.ims[add_idx])
                ptr += 1
                if ptr == len(cams):
                    ptr = 1
                n_now += 1
            ids = ids + add_ids
            cams = np.concatenate([cams,np.stack(add_cams,axis=0)],axis=0)
            cams = np.ascontiguousarray(cams)
            ims = np.concatenate([ims,np.stack(add_ims,axis=0)],axis=0)
            ims = np.ascontiguousarray(ims)


        py_depths = self.py_depths
        py_costs = self.py_costs
        py_planeHypos = self.py_planeHypos
        py_vselects = self.py_vselects
        ncc_module.init(cams.data, ims.data, py_depths.data, py_costs.data, py_planeHypos.data, py_vselects.data)
        ncc_module.set_init_flag(True)
        ncc_module.set_init_flag_norand(False)
        ncc_module.genCamFromNp()
        ncc_module.dataToCuda()
        ncc_module.ProcessCuNcc(0)
        ncc_module.set_init_flag(False)
        ncc_module.set_init_flag_norand(False)
        ncc_module.ProcessCuNcc(0)
        ncc_module.ProcessCuNcc(0)
        ncc_module.ProcessCuNcc(0)
        ncc_module.ProcessCuNcc(0)
        # ncc_module.ProcessCuNcc(0)
        # ncc_module.ProcessCuNcc(0)
        self.cost[:,:,0] = ncc_module.bind_ncc_cu()[0, :, :]
        self.hypos[:,:,:] = ncc_module.bind_hypos_cu()[0, :, :]

        # ncc_module.CalNcc(0, self.coord_tem, self.coord_num_tem)
        #
        # ncc_module.bind_hypos_cu()[0, :, :] = torch.tensor(self.priors[idx],dtype=torch.float32, device='cuda')
        #
        # plt.imshow(ncc_module.bind_ncc_cu()[0, :, :].cpu())
        # # plt.imshow(self.priors[9][:, :, 1:] * 0.5 + 0.5)
        # # ncc_module.bind_ncc_cu()[0, :, :] = ncc_module.bind_ncc_cu()[1, :, :]
        # plt.show()

    def process_with_prior(self, idx):
        # usage
        # initialization
        ncc_module = ncc.pyNCCmodule()
        # ncc_module.__version__()
        ncc_module.set_hw(self.h, self.w)
        ncc_module.set_set_pert_depth_ratio(self.helix_ratio)
        # load the pair. info
        ncc_module.load_pair(self.fake_pair_path)
        ids = [idx] + self.pairs_dict[idx]
        cams = self.cams[ids]
        ims = self.ims[ids]

        if len(ids) < self.n_buffer:
            print(f'warning! src imgs for {idx} not enough!')
            n_now = len(ids)
            add_cams = []
            add_ims = []
            add_ids = []
            ptr = 1
            while n_now < self.n_buffer:
                add_idx = ids[ptr]
                add_ids.append(add_idx)
                add_cams.append(self.cams[add_idx])
                add_ims.append(self.ims[add_idx])
                ptr += 1
                if ptr == len(cams):
                    ptr = 1
                n_now += 1
            ids = ids + add_ids
            cams = np.concatenate([cams,np.stack(add_cams,axis=0)],axis=0)
            cams = np.ascontiguousarray(cams)
            ims = np.concatenate([ims,np.stack(add_ims,axis=0)],axis=0)
            ims = np.ascontiguousarray(ims)

        py_depths = self.py_depths
        py_costs = self.py_costs
        py_planeHypos = self.py_planeHypos
        py_vselects = self.py_vselects
        ncc_module.init(cams.data, ims.data, py_depths.data, py_costs.data, py_planeHypos.data, py_vselects.data)
        ncc_module.set_init_flag(True)
        ncc_module.genCamFromNp()
        ncc_module.dataToCuda()
        ncc_module.set_init_flag(False)
        ncc_module.set_init_flag_norand(True)
        ncc_module.bind_hypos_cu()[0, :, :,:] = torch.tensor(self.priors[idx], dtype=torch.float32).cuda()
        ncc_module.ProcessCuNcc(0)
        ncc_module.set_init_flag(False)
        ncc_module.set_init_flag_norand(False)
        ncc_module.ProcessCuNcc(0)
        ncc_module.ProcessCuNcc(0)
        ncc_module.ProcessCuNcc(0)
        self.cost[:, :, 0] = ncc_module.bind_ncc_cu()[0, :, :]
        self.hypos[:, :, :] = ncc_module.bind_hypos_cu()[0, :, :]

    def process_with_prior_manual(self, idx, prior_manual):
        # usage
        # initialization
        ncc_module = ncc.pyNCCmodule()
        # ncc_module.__version__()
        ncc_module.set_hw(self.h, self.w)
        ncc_module.set_set_pert_depth_ratio(self.helix_ratio)
        # load the pair. info
        ncc_module.load_pair(self.fake_pair_path)
        ids = [idx] + self.pairs_dict[idx]
        cams = self.cams[ids]
        ims = self.ims[ids]


        if len(ids) < self.n_buffer:
            print(f'warning! src imgs for {idx} not enough!')
            n_now = len(ids)
            add_cams = []
            add_ims = []
            add_ids = []
            ptr = 1
            while n_now < self.n_buffer:
                add_idx = ids[ptr]
                add_ids.append(add_idx)
                add_cams.append(self.cams[add_idx])
                add_ims.append(self.ims[add_idx])
                ptr += 1
                if ptr == len(cams):
                    ptr = 1
                n_now += 1
            ids = ids + add_ids
            cams = np.concatenate([cams,np.stack(add_cams,axis=0)],axis=0)
            cams = np.ascontiguousarray(cams)
            ims = np.concatenate([ims,np.stack(add_ims,axis=0)],axis=0)
            ims = np.ascontiguousarray(ims)


        py_depths = self.py_depths
        py_costs = self.py_costs
        py_planeHypos = self.py_planeHypos
        py_vselects = self.py_vselects
        ncc_module.init(cams.data, ims.data, py_depths.data, py_costs.data, py_planeHypos.data, py_vselects.data)
        ncc_module.set_init_flag(True)
        ncc_module.genCamFromNp()
        ncc_module.dataToCuda()
        ncc_module.set_init_flag(False)
        ncc_module.set_init_flag_norand(True)
        ncc_module.bind_hypos_cu()[0, :, :,:] = torch.tensor(prior_manual, dtype=torch.float32).cuda()
        ncc_module.ProcessCuNcc(0)
        self.cost[:, :, 0] = ncc_module.bind_ncc_cu()[0, :, :]
        self.hypos[:, :, :] = ncc_module.bind_hypos_cu()[0, :, :]

    def set_pert_ratio(self, pert):
        self.helix_ratio = max(pert,0.005)

    def load_mask(self):
        if self.maskflag:
            for i in range(self.n_image):
                mask = cv2.imread(self.meta['maskpaths'][i], -1)
                if self.meta['resolution'] != 1:
                    mask = cv2.resize(mask, (mask.shape[1]//self.meta['resolution'], mask.shape[0]//self.meta['resolution']))
                if len(mask.shape) == 3:
                    mask = mask[:, :, -1] > 0
                self.masks.append(mask)

    def depth_reproj_filter(self):
        # prepare data
        projs = []
        for i in tqdm.tqdm(range(self.n_image)):
            intr_mat = self.cams[i][0][:3, :3]
            extr_mat = self.cams[i][1][:3,:4]
            proj_mat = np.eye(4)
            proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
            projs.append(torch.from_numpy(proj_mat))
        depths = self.depths
        confs = self.costs

        depths = torch.stack(depths).float().cuda()
        projs = torch.stack(projs).float().cuda()
        confs = torch.stack(confs).float().cuda()

        self.filter_depth_mask = []

        for i in tqdm.tqdm(range(self.n_image)):
            ids = [i] + self.pairs_dict[i]
            projs_case = projs[ids]
            depths_case = depths[ids]
            confs_case = confs[ids]

            pc_buff = torch.zeros((3, self.h, self.w), device=depths_case.device, dtype=depths_case.dtype)
            val_cnt = torch.zeros((1, self.h, self.w), device=depths_case.device, dtype=depths_case.dtype)


            ref_pc, pcs, dist = filter_depth(ref_depth=depths_case[0:1],
                                             src_depths=depths_case[1:],
                                             ref_proj=projs_case[0:1],
                                             src_projs=projs_case[1:])
            masks = (dist < self.depth_filter_param['dist_thresh']).float()
            masked_pc = pcs * masks
            pc_buff += masked_pc.sum(dim=0, keepdim=False)
            val_cnt += masks.sum(dim=0, keepdim=False)

            final_mask = (val_cnt >= self.depth_filter_param['num_consist']).squeeze(0) * (confs_case[0].squeeze(-1) < self.depth_filter_param['prob_thresh'])
            self.filter_depth_mask.append(final_mask.cpu())