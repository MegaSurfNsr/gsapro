#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os, sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())
import glob
from PIL import Image
import torch
from scene import Scene
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import trimesh, copy
from collections import deque


def load_cam(file, interval_scale=1):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    words = open(file).read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 256
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def load_prior_mask(path):
    ext = path.split('.')[-1]
    if ext == 'png':
        mask = cv2.imread(path, -1)
        if len(mask.shape) == 3:
            mask = mask[:, :, -1] > 0
    else:
        raise NotImplementedError
    return mask

def parse_cameras(path):
    cam = load_cam(path)
    extr_mat = cam[0]
    intr_mat = cam[1]

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat

def load_data(cam_path, helixout_path, image_path, prior_mask=None,rgb_flag=False):
    '''

    :param root_path:
    :param scene_name:
    :param thresh:
    :return: depth
    '''
    rgb_paths = sorted(glob.glob(os.path.join(image_path, '*')))
    cam_paths = sorted(glob.glob(os.path.join(cam_path, '*')))
    helixout_paths = sorted(glob.glob(os.path.join(helixout_path, '*')))

    if prior_mask is not None:
        prior_mask_paths = sorted(glob.glob(os.path.join(prior_mask, '*')))

    depths = []
    prior_masks = []
    projs = []
    rgbs = []
    confs = []
    norms = []
    intrins = []
    extrins = []
    near_fars = []

    if len(rgb_paths) == len(helixout_paths):
        args.prefix = 'nocost_'
    for i in tqdm(range(len(rgb_paths))):
        extr_mat, intr_mat = parse_cameras(cam_paths[i])
        if args.scale != 1:
            intr_mat[:2, :] = intr_mat[:2, :] / args.scale
        intrins.append(intr_mat[:3, :3])
        extrins.append(extr_mat)
        near_fars.append(intr_mat[3,:])
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))

        if len(rgb_paths) == len(helixout_paths):
            helixout = np.load(helixout_paths[i])
            if helixout.shape[0] <=10:
                helixout = helixout.transpose(1,2,0)
            if len(helixout.shape)==2:
                helixout = helixout[:,:,None]
            conf_map = np.ones((helixout.shape[0], helixout.shape[1], 1), dtype=np.float32) * 0.2
        else:
            conf_map = np.load(helixout_paths[i])
            helixout = np.load(helixout_paths[i + len(rgb_paths)])
            if helixout.shape[0] <=10:
                helixout = helixout.transpose(1,2,0)
            if len(helixout.shape)==2:
                helixout = helixout[:,:,None]
            if conf_map.shape[0] <=10:
                conf_map = conf_map.transpose(1,2,0)
            if len(conf_map.shape) ==2:
                conf_map = conf_map[:,:,None]


        h, w, _ = conf_map.shape
        confs.append(torch.from_numpy(conf_map))

        h_idx = np.arange(h, dtype=np.int32)
        w_idx = np.arange(w, dtype=np.int32)
        raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
        raw_grid.append(np.ones([h, w]))
        dep_map = helixout
        depths.append(torch.from_numpy(dep_map))

        if rgb_flag:
            img_raw = cv2.imread(rgb_paths[i], -1)
            img_raw = cv2.resize(img_raw, (helixout.shape[1], helixout.shape[0]))
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            img_raw = o3d.geometry.Image(img_raw)
            rgbs.append(img_raw)
        if prior_mask is not None:
            prior_raw = load_prior_mask(prior_mask_paths[i])
            prior_raw = cv2.resize(np.asarray(prior_raw,dtype=np.uint8), (helixout.shape[1], helixout.shape[0])) > 0.5

            prior_masks.append(prior_raw)
    # plt.imshow(dep_map)
    # plt.show()

    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()
    confs = torch.stack(confs).float()
    if prior_mask is not None:
        prior_masks = torch.from_numpy(np.stack(prior_masks)).float()

    if args.device == 'cuda' and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()
        confs = confs.cuda()
        if prior_mask is not None:
            prior_masks = prior_masks.cuda()
    return depths, projs, rgbs, confs, norms, prior_masks, intrins, extrins, near_fars



if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Depth fusion with consistency check.')
    parser.add_argument('--cam_path', type=str, default='')
    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--depth_path', type=str, default='')
    parser.add_argument('--prior_mask', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--auto_voxel', action='store_true')
    parser.add_argument('--near_far_clip', action='store_true')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--voxel_size', type=float, default=0.002)
    parser.add_argument('--max_depth', type=float, default=5)

    args = parser.parse_args()

    depths, projs, rgbs, confs, norms, prior_masks, intrins, extrins, near_fars = load_data(args.cam_path, args.depth_path, args.image_path,
                                                               args.prior_mask,rgb_flag=True)

    near_fars = np.asarray(near_fars)
    voxel_size = args.voxel_size
    sdf_trunc_size = 8.0 * voxel_size #15
    if args.auto_voxel:
        campcd = []
        for extrin in extrins:
            pose = np.linalg.pinv(extrin)
            campcd.append(np.matmul(pose,np.asarray([0,0,0,1])))
        campcd = np.asarray(campcd)[:,:3]
        maxbound = campcd.max(axis=0)
        minbound = campcd.min(axis=0)
        extent = maxbound - minbound
        voxel_size = extent.max().item() / 2048
        args.max_depth = extent.max()
        sdf_trunc_size =15.0 * voxel_size

    if args.near_far_clip:
        far = near_fars.max().item()
        args.max_depth = far
        sdf_trunc_size = 15.0 * voxel_size

    print(f"TSDF voxel_size {voxel_size}")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


    for idx in tqdm(range(len(depths))):

        ref_depth = depths[idx]
        if args.prior_mask is not None:
            ref_depth[torch.logical_not(prior_masks[idx].bool())] = 0
        # ref_depth[ref_depth > args.max_depth] = 0
        if args.near_far_clip:
            viewmask =(ref_depth > near_fars[idx][0]) * (ref_depth < near_fars[idx][3])
            ref_depth[torch.logical_not(viewmask)] = 0
        ref_depth = ref_depth.detach().cpu().numpy()[:,:,0]

        pose = np.linalg.pinv(extrins[idx])
        extrin = extrins[idx]

        color = rgbs[idx]
        if np.asarray(color).shape[2] == 4:
            image_array_rgb = np.asarray(color)[:, :, :3]
            color = o3d.geometry.Image(np.ascontiguousarray(image_array_rgb))

        depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))

        H, W = ref_depth.shape
        Fx = intrins[idx][0,0]
        Fy = intrins[idx][1,1]
        Cx = intrins[idx][0,2]
        Cy = intrins[idx][1,2]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=args.max_depth, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(W, H, Fx, Fy, Cx, Cy),
            extrin)

    mesh = volume.extract_triangle_mesh()

    os.makedirs(args.save_path, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(args.save_path, "tsdf_fusion.ply"), mesh,
                               write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

# --cam_path /mnt/data2/yswang2024/gaussian_surfels_data/2dgs/DTU/scan24/cams --image_path /mnt/data2/yswang2024/gaussian_surfels_data/2dgs/DTU/scan24/images --helixout_path /mnt/data4/yswangdata4/experiments/dtu_gs2d/gs2d_scan24/gs2d_it30000 --save_path /mnt/data4/yswangdata4/experiments/dtu_gs2d/gs2d_scan24/helix_out --prior_mask /mnt/data2/yswang2024/gaussian_surfels_data/2dgs/DTU/scan24/images

