import os
import argparse
import numpy as np
import json
import shutil
import open3d as o3d
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--hw_root', type=str, default='/mnt/data3/yswang2024_data3/dataset/hw_polytech/yusen_ploytech_opendata')
parser.add_argument('--out', type=str, default='/mnt/data3/yswang2024_data3/dataset/hw_polytech/processed_dataset')
args = parser.parse_args()

if __name__ == '__main__':
    blocks_root = os.path.join(args.hw_root, 'sdf_studio_simple')
    blocks = sorted(os.listdir(blocks_root))
    # sfm = o3d.io.read_point_cloud(os.path.join(args.hw_root,'sfm.ply'))
    for block in blocks:
        blockroot = os.path.join(blocks_root, block)
        os.makedirs(os.path.join(args.out, block), exist_ok=True)
        os.makedirs(os.path.join(args.out, block, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out, block, 'cams'), exist_ok=True)
        os.makedirs(os.path.join(args.out, block, 'masks'), exist_ok=True)
        shutil.copyfile(os.path.join(blockroot,'pair.txt'), os.path.join(args.out, block, 'pair.txt'))
        shutil.copyfile(os.path.join(args.hw_root,'sfm.ply'), os.path.join(args.out, block, 'points.ply'))


        with open(os.path.join(blockroot,'transforms.json'), 'r') as f:
            taskdict = json.load(f)

        # worldtogt = np.asarray(taskdict['worldtogt'])
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector((sfm.points - worldtogt[:3,3]) / worldtogt[0,0] )
        # o3d.io.write_point_cloud(os.path.join(args.out, block, 'points.ply'), pcd)

        newid = 0
        for frame in taskdict['frames']:
            maskpath = os.path.join(args.hw_root,'grid_new',block,os.path.basename(frame['mask_path']))
            filepath = os.path.join(args.hw_root,'images',os.path.basename(frame['file_path']))
            basename = os.path.basename(filepath).split('.')[0]
            campath = os.path.join(args.hw_root,'cams', basename + '_cam.txt')
            if filepath.endswith('.jpg'):
                shutil.copyfile(filepath, os.path.join(args.out, block, 'images',int(newid).__str__().zfill(8)+'.jpg' ))
            shutil.copyfile(maskpath, os.path.join(args.out, block, 'masks', int(newid).__str__().zfill(8) + '.png'))
            shutil.copyfile(campath, os.path.join(args.out, block, 'cams', int(newid).__str__().zfill(8) + '_cam.txt'))
            newid = newid + 1

        # source = f"python /home/yswang/data3/code/PGSR_main/minor_scripts/acmh2colmap.py --cams {os.path.join(args.out, block, 'cams')} --img_folder {os.path.join(args.out, block, 'images')} --out {os.path.join(args.out, block)}"
        # os.system(source)
        # source = f"cp {os.path.join(args.out, block,'sparse','0','*')} {os.path.join(args.out, block,'sparse')}"
        # os.system(source)