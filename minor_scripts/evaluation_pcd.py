import os.path
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import argparse
import errno, os
import glob
import os.path as osp
import re
# import matplotlib.pyplot as plt
import cv2
from PIL import Image
import gc
import tqdm
import open3d as o3d


parser = argparse.ArgumentParser(description='eval point cloud.')
parser.add_argument('--pcd', type=str, default=None)
parser.add_argument('--gt', type=str, default=None)
parser.add_argument('--voxelsize', type=float, default=0.05)
parser.add_argument('--exclude_distance', type=float, default=None)
parser.add_argument('--downvoxel', action='store_true')
parser.add_argument('--f1thred', type=float, default=0.2)
parser.add_argument('--outf', type=str, default='/mnt/data3/yswang2024_data3/test.txt')
args = parser.parse_args()
def evaluation_mesh(gtn,pcn,voxel_size=0.025,thred=0.2,downvoxel=True, exclude_distance=None):
    print(f'reading {gtn}')
    pcd_gt = o3d.io.read_point_cloud(gtn)
    if downvoxel:
        print('down voxel')
        pcd_gt = pcd_gt.voxel_down_sample(voxel_size=voxel_size)
    # pcd_gt.paint_uniform_color((1,0,0))
    print(f'reading {pcn}')

    pcd_check = o3d.io.read_point_cloud(pcn)
    if downvoxel:
        print('down voxel')
        pcd_check = pcd_check.voxel_down_sample(voxel_size=voxel_size)  # 0.1
    print(f'evaluation process')
    dists_c2gt = np.asarray(pcd_check.compute_point_cloud_distance(pcd_gt))
    dists_gt2c = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_check))

    precision = 100*(dists_c2gt <= thred).sum()/dists_c2gt.shape[0]
    recall = 100*(dists_gt2c <= thred).sum()/dists_gt2c.shape[0]

    precision2 = 100*(dists_c2gt <= thred/2).sum()/dists_c2gt.shape[0]
    recall2 = 100*(dists_gt2c <= thred/2).sum()/dists_gt2c.shape[0]


    f_score = 2*precision*recall/(precision+recall)
    f_score2 = 2*precision2*recall2/(precision2+recall2)

    accuracy = dists_c2gt.mean()
    completeness = dists_gt2c.mean()


    print(f'accuracy {accuracy}')
    print(f'completeness {completeness}')
    print(f'precision {precision}')
    print(f'recall {recall}')
    print(f'fscore {f_score}')
    print(f'precision2 {precision2}')
    print(f'recall2 {recall2}')
    print(f'fscore2 {f_score2}')
    print('\n')

    if exclude_distance is not None:
        pcd_check.points = o3d.utility.Vector3dVector(np.asarray(pcd_check.points)[dists_c2gt<exclude_distance,:])
        pcd_gt.points = o3d.utility.Vector3dVector(np.asarray(pcd_gt.points)[dists_gt2c < exclude_distance, :])

        dists_c2gt = np.asarray(pcd_check.compute_point_cloud_distance(pcd_gt))
        dists_gt2c = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_check))

        precision_exc = 100*(dists_c2gt <= thred).sum()/dists_c2gt.shape[0]
        recall_exc = 100*(dists_gt2c <= thred).sum()/dists_gt2c.shape[0]

        precision2_exc = 100*(dists_c2gt <= thred/2).sum()/dists_c2gt.shape[0]
        recall2_exc = 100*(dists_gt2c <= thred/2).sum()/dists_gt2c.shape[0]


        f_score_exc = 2*precision_exc*recall_exc/(precision_exc+recall_exc)
        f_score2_exc = 2*precision2_exc*recall2_exc/(precision2_exc+recall2_exc)

        accuracy_exc = dists_c2gt.mean()
        completeness_exc = dists_gt2c.mean()

        return (precision,recall,f_score,precision2,recall2,f_score2,accuracy, completeness, (accuracy+completeness) /2,
                precision_exc,recall_exc,f_score_exc,precision2_exc,recall2_exc,f_score2_exc,accuracy_exc, completeness_exc, (accuracy_exc+completeness_exc) /2)
    else:
        return (precision,recall,f_score,precision2,recall2,f_score2,accuracy, completeness, (accuracy+completeness) /2,
                0,0,0,0,0,0,0, 0, (0+0) /2)

if __name__ == '__main__':
    gtn = os.path.join(args.gt)
    pcn = os.path.join(args.pcd)
    r = evaluation_mesh(gtn, pcn,voxel_size=args.voxelsize,thred=args.f1thred,downvoxel=args.downvoxel,exclude_distance=args.exclude_distance)
    os.makedirs(os.path.dirname(args.outf), exist_ok=True)
    r = tuple([float(x) for x in r])

    with open(os.path.join(args.outf), 'a') as f:
        f.write(f'pcd: {args.pcd}\n')
        f.write(f'gt: {args.gt}\n')
        f.write(f'precision recall f_score precision2 recall2 f_score2 accuracy completeness overall precision_exc recall_exc f_score_exc precision2_exc recall2_exc f_score2_exc accuracy_exc completeness_exc overall_exc\n')
        f.write(f'{r}\n')


#
# m1 = o3d.io.read_point_cloud('/home/yswang/server160/yswang_code/neus1214/exp/s0785_ap7/womask_sphere/meshes/full_resolution.ply')
# cl, ind = m1.remove_statistical_outlier(nb_neighbors=20,
#                                                     std_ratio=2.0)
# o3d.visualization.draw_geometries([m1])
