import cv2
import numpy as np
import tqdm
import open3d as o3d
import tqdm
import os
import numpy as np
import glob
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert colmap camera')

    parser.add_argument('--mesh', type=str, help='Project dir.')
    parser.add_argument('--bbox', type=str, help='Project dir.')
    parser.add_argument('--bbox_all', type=str, help='Project dir.')
    parser.add_argument('--out', type=str, help='Project dir.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)

    if args.bbox_all is not None:
        print(args.bbox_all)
        bbox_all = o3d.io.read_point_cloud(args.bbox_all)
        obb_all = bbox_all.get_oriented_bounding_box()
        mesh = mesh.crop(obb_all)
    print(args.bbox)
    bbox = o3d.io.read_point_cloud(args.bbox)
    obb = bbox.get_oriented_bounding_box()
    clip_mesh = mesh.crop(obb)
    o3d.io.write_triangle_mesh(args.out,clip_mesh)