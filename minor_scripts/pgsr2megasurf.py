import open3d as o3d
import numpy as np
import xmltodict
import argparse
import ysutils
import os
import tqdm
import glob
import cv2
import ysutils
import math
import json

def generate_json(args):
    cams_file = glob.glob(os.path.join(args.cams,'*'))
    cams_file.sort()
    # check id consistency
    poses = []
    intrins = []
    extrins = []
    for cam_f in cams_file:
        cam = ysutils.util_mvsnet.load_cam(cam_f)
        intrin_info = cam[1]
        intrin = intrin_info[:3,:3]
        extrins.append(cam[0])
        pose = np.linalg.inv(cam[0])
        poses.append(pose)
        intrins.append(intrin)


    # read all tiepoints
    pcd = o3d.io.read_point_cloud(args.ply)
    meta_aabb: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd)
    # radius = meta_aabb.get_extent().min()/50
    # pcd_clean = pcd.remove_radius_outlier(20,radius)[0]
    pcd_clean = pcd
    meta_aabb :o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd_clean)
    cube, aabb = ysutils.util_megasurf.get_bbox_from_pcd(pcd_clean)
    o3d.io.write_triangle_mesh(os.path.join(args.out, f'realworld_bbox.ply'), cube)
    o3d.io.write_point_cloud(os.path.join(args.out, f'clen_tiepoints_local.ply'), pcd_clean)

    # calculate scale mat
    scale_mat = np.eye(4)
    cube_cal, bbox_cal = ysutils.util_megasurf.get_bbox_from_pcd(pcd_clean)
    scale = bbox_cal.get_extent().max() / 2
    meanvec = bbox_cal.get_center()
    x_mean, y_mean, z_mean = meanvec
    scale_mat[:3, :3] = scale_mat[:3, :3] * scale
    scale_mat[:3, 3] = meanvec
    cube_cal.vertices = o3d.utility.Vector3dVector((cube_cal.vertices - meanvec) /scale)
    pcd_clean.points = o3d.utility.Vector3dVector((pcd_clean.points - meanvec) /scale)
    o3d.io.write_triangle_mesh(os.path.join(args.out, "normalized_bbox.ply"),cube_cal)
    o3d.io.write_point_cloud(os.path.join(args.out, "normalized_points.ply"),pcd_clean)

    # casedict = {'acmh_root':'''/mnt/data2/yswang2024/dataset/scan24_2dgs''',
    #             'case_images':('''/mnt/data2/yswang2024/dataset/scan24_2dgs/images_acmh/''','','''00000000''','.jpg'),
    #             'case_mask':('''/mnt/data2/yswang2024/dataset/dtu24_acmh/block_mask/''','blockmask_','00000000','.jpg'),
    #             'case_confidence_mask':('''/mnt/data2/yswang2024/dataset/dtu24_acmh/filtered/mask/''','','00000000','_geo.png'),
    #             'h':1162,
    #             'w':1554}

    recs = []

    imgpaths = glob.glob(os.path.join(args.images,'*.jpg'))
    imgpaths.sort()
    depthpaths = glob.glob(os.path.join(args.depths,'hypo_*.npy'))
    depthpaths.sort()

    tempimg = cv2.imread(imgpaths[0])
    h,w,_ = tempimg.shape

    if args.masks is not None:
        maskpaths = glob.glob(os.path.join(args.masks,'*.png'))
        maskpaths.sort()

    for i in tqdm.tqdm(range(len(cams_file))):
        imgpath = imgpaths[i]
        depthpath = depthpaths[i]
        costpath = ''
        if args.masks is not None:
            maskpath = maskpaths[i]
        else:
            maskpath = ''
        confpath = ''

        img_rec = {
            'file_path': imgpath,
            'depth_file': depthpath,
            'cost_file': costpath,
            'mask_path': maskpath,
            'confidence_mask': confpath,
            'intrinsic': intrins[i][:3, :3],
            'pose': poses[i],
            'extrin':extrins[i],
            'h': h,
            'w': w
        }
        recs.append(img_rec)

    # generate json
    fix_size_flag = True

    out = {
        "coord_sys":  "OPENGL",
        "is_fisheye": False,
        "world_to_gt": scale_mat.tolist(),
        "n_frames": len(cams_file),
    }
    if fix_size_flag:
        out.update({
            "w": int(w),
            "h": int(h),
        })


    frames = []
    for idx, rec in enumerate(recs):
        intrin = np.eye(4)
        intrin[:3, :3] = rec['intrinsic']
        extrin = rec['extrin']
        proj_mat = intrin @ extrin

        # scale and decompose
        P = proj_mat @ scale_mat
        P = P[:3, :4]
        intrinsic_param, c2w = ysutils.util_neuralangelo.load_K_Rt_from_P(None, P)
        c2w_gl = ysutils.util_neuralangelo._cv_to_gl(c2w)
        angle_x = math.atan(w / (intrinsic_param[0][0] * 2)) * 2
        angle_y = math.atan(h / (intrinsic_param[1][1] * 2)) * 2

        # 'file_path': imgpath,
        # 'depth_file': depthpath,
        # 'cost_file': costpath,
        # 'mask_path': maskpath,
        # 'confidence_mask': confpath,
        # 'intrinsic': intrins[i][:3, :3],
        # 'pose': poses[i],
        # 'extrin': extrins[i],
        # 'h': casedict['h'],
        # 'w': casedict['w']
        #
        frame = {"file_path": rec['file_path'],
                 "depth_file": rec['depth_file'],
                 "cost_file": rec['cost_file'],
                 "mask_path": rec['mask_path'],
                 "confidence_mask": rec['confidence_mask'],
                 "transform_matrix": c2w_gl.tolist(),
                 "fl_x": intrinsic_param[0][0],
                 "fl_y": intrinsic_param[1][1],
                 "cx": intrinsic_param[0][2],
                 "cy": intrinsic_param[1][2],
                 "camera_angle_x": angle_x,
                 "camera_angle_y": angle_y,
                 }

        frames.append(frame)
    out.update({
        "frames": frames
    })

    file_path = os.path.join(args.out, 'transforms.json')
    with open(file_path, "w") as outputfile:
        json.dump(out, outputfile, indent=2)



if __name__ == '__main__':
    print("only for test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--cams', type=str, default="""/mnt/data3/yswang2024_data3/dataset/blended_downsample/5af02e904c8216544b4ab5a2/cams""")
    parser.add_argument('--images', type=str, default="""/mnt/data3/yswang2024_data3/dataset/blended_downsample/5af02e904c8216544b4ab5a2/images""")
    parser.add_argument('--depths', type=str, default="""/mnt/data3/yswang2024_data3/helix_output/helixonlyncc_5af02e904c8216544b4ab5a2/helix_out/pgsr_it30000""")
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--ply', type=str, default="""/mnt/data3/yswang2024_data3/dataset/blended_mesh/5af02e904c8216544b4ab5a2.ply""")
    parser.add_argument('--out', type=str, default="""/mnt/data3/yswang2024_data3/dataset/blended_downsample/5af02e904c8216544b4ab5a2""")
    args = parser.parse_args()
    generate_json(args)