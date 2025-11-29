import argparse
import os.path

import sys
sys.path.append(os.getcwd())
sys.path.append('../')
import os
import ysutils
import json
import numpy as np
from ysutils.util_colmap import fake_colmap
import glob
import cv2


if __name__ == '__main__':
    print("only for test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cams', type=str, default=None)
    parser.add_argument('--img_folder', type=str, default=None)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--cam_downscale', type=int, default=1)


    args = parser.parse_args()
    print(args.cams)
    # print(args.img_folder)
    args.out = os.path.join(args.out,'sparse','0')
    os.makedirs(args.out, exist_ok=True)

    camlist = glob.glob(os.path.join(args.cams, '*cam.txt'))
    imglist = glob.glob(os.path.join(args.img_folder, '*'))
    print(len(camlist), len(imglist))


    camlist.sort()
    imglist.sort()

    extrins = []
    intrins = []
    image_filenames = []
    temimg = cv2.imread(imglist[0])
    h, w, _ = temimg.shape
    height = []
    width = []

    for i in range(len(camlist)):
        cams = ysutils.util_mvsnet.load_cam(camlist[i])
        extrins.append(cams[0])
        intrin = cams[1][:3,:3]
        intrin[:2,:] = intrin[:2,:] / args.cam_downscale
        intrins.append(intrin)
        image_filenames.append(os.path.basename(imglist[i]))
        height.append(h)
        width.append(w)


    fake_colmap(image_filenames,height, width, intrins, extrins, args.out)

    # # results check
    # import open3d as o3d
    # tpoints = o3d.io.read_point_cloud("F:\yusen_ploytech_opendata\yusen_ploytech_opendata\sdf_studio_simple\\4\\sfm_rescale.ply")
    # cam_pcd = o3d.geometry.PointCloud()
    # from ysutils.util_open3d import gen_cam_frust
    # poses = [np.linalg.pinv(extrin) for extrin in extrins]
    # poses_gl = [ysutils.util_neuralangelo._gl_to_cv(pose) for pose in poses]
    #
    # cams = gen_cam_frust(poses, 0.1)
    # cams_gl = gen_cam_frust(poses_gl, 0.1)
    #
    # o3d.visualization.draw_geometries([tpoints] + cams + cams_gl)
    # o3d.visualization.draw_geometries([tpoints] + cams_gl)

#     print(' ')
#
# import shutil
# l = glob.glob('/mnt/data4/yswangdata4/dataset/fake_2dgs_scan63/images/*')
# l.sort()
# for i in range(len(l)):
#     shutil.copyfile(l[i], os.path.join('/mnt/data4/yswangdata4/dataset/fake_2dgs_scan63/images',f'{i.__str__().zfill(4)}.png'))
#
