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
    root = '/mnt/data3/yswang2024_data3/dataset/blended_downsample/'
    scene_file = '/mnt/data3/yswang2024_data3/dataset/blended_highres/process_scene2'
    with open(scene_file,'r') as f:
        scenelist = f.readlines()
    scenelist = [s.strip() for s in scenelist]


    for scene in scenelist:
        scene_root = os.path.join(root,scene)
        outpath = os.path.join(scene_root,'sparse','0')
        os.makedirs(outpath, exist_ok=True)

        camlist = glob.glob(os.path.join(scene_root,'cams', '*cam.txt'))
        imglist = glob.glob(os.path.join(scene_root,'images', '*'))

        if len(imglist) != len(camlist):
            continue

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
            intrins.append(intrin)
            image_filenames.append(os.path.basename(imglist[i]))
            height.append(h)
            width.append(w)


        fake_colmap(image_filenames,height, width, intrins, extrins, outpath)

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

    print(' ')




