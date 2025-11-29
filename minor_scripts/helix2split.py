import open3d as o3d
import numpy as np
import os
import argparse
import glob
def get_bbox_from_pcd(pcd, clip=False, min_len=0.001,scale=1.1):
    # obb = pcd.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    aabb = pcd.get_axis_aligned_bounding_box()
    # aabb.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd, aabb])
    obb = aabb.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)
    if clip:
        obb.extent = obb.extent * scale
        obb.extent = obb.extent.clip(min=min_len)
    aabb2 = obb.get_axis_aligned_bounding_box()
    # aabb2.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd, aabb2])
    cube = o3d.geometry.TriangleMesh.create_box(width=aabb2.get_extent()[0],height=aabb2.get_extent()[1],depth=aabb2.get_extent()[2])
    cube.translate(
        aabb2.get_center(),
        relative=False,
    )
    # o3d.visualization.draw_geometries([cube, aabb2])
    return cube, aabb2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/mnt/data3/yswang2024_data3/dataset/blended_downsample')
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--ply', type=str, default=None)
    parser.add_argument('--bboxroot', type=str, default='/mnt/data3/yswang2024_data3/dataset/blended_mesh/split')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out), exist_ok=True)
    sfm = o3d.io.read_point_cloud(os.path.join(args.dataroot, args.scene, 'points.ply'))
    tem = glob.glob(os.path.join(args.bboxroot, '*.ply'))
    splits_bbox = []
    for b in tem:
        if os.path.basename(b).split('_')[1] == 'bbox' and os.path.basename(b).split('_')[0] == args.scene:
            splits_bbox.append(b)

    pcd_clean = o3d.io.read_point_cloud(args.ply)
    pcd_clean.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_clean.points), np.asarray(sfm.points)],axis=0))
    pcd_clean.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_clean.colors), np.zeros_like(sfm.points)],axis=0))

    # make output folders
    for i in range(len(splits_bbox)):
        bbox = o3d.io.read_point_cloud(splits_bbox[i])
        savename = os.path.basename(splits_bbox[i]).split('.')[0].split('_')
        savename = savename[0] + '_' + savename[2]
        meta_aabb: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(bbox)
        # tie points within the bbox
        croppcd = pcd_clean.crop(meta_aabb)

        aabb = croppcd.get_axis_aligned_bounding_box()

        ratio = min(6_000_000 / len(croppcd.points), 1)
        # print(ratio)
        if ratio >= 0.9999:
            o3d.io.write_point_cloud(os.path.join(args.out, f'{savename}.ply'), croppcd)
            continue
        print('downsampling')
        dpcd = croppcd.random_down_sample(ratio)
        # dpcd = dpcd.voxel_down_sample(voxel_size=0.025)
        o3d.io.write_point_cloud(os.path.join(args.out, f'{savename}.ply'), dpcd)

