import open3d as o3d
import numpy as np
import os
import argparse
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
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--ply', type=str, default=None)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--prefix', type=str, default='')

    args = parser.parse_args()

    pcd_clean = o3d.io.read_point_cloud(args.ply)
    meta_aabb: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.PointCloud.get_axis_aligned_bounding_box(pcd_clean)
    block_length = meta_aabb.get_extent() / args.p #* (1 + overlap_side_length)

    nblocks = np.ceil(meta_aabb.get_extent() / (block_length + 0.1))[:2]



    x_coord = (np.arange(0, nblocks[0])) * block_length[0]
    y_coord = (np.arange(0, nblocks[1])) * block_length[1]
    x_coord = x_coord - x_coord.mean()
    y_coord = y_coord - y_coord.mean()

    splits_bbox = []
    for x_c in x_coord:
        for y_c in y_coord:
            split_bbox_min = [x_c - block_length[0] * 1.1 / 2, y_c - block_length[1] * 1.1 / 2,
                              meta_aabb.min_bound[-1]]
            split_bbox_max = [x_c + block_length[0] * 1.1 / 2, y_c + block_length[0] * 1.1 / 2,
                              meta_aabb.max_bound[-1]]
            splits_bbox.append(np.asarray(split_bbox_min + split_bbox_max))
    os.makedirs(os.path.join(args.out), exist_ok=True)
    np.savetxt(os.path.join(args.out, f'{args.prefix}_splitsbbox.txt'), splits_bbox)

    # make output folders
    for i in range(len(splits_bbox)):
        tempcd = o3d.geometry.PointCloud()
        tempcd.points = o3d.utility.Vector3dVector(splits_bbox[i].reshape(-1, 3))
        cube, aabb = get_bbox_from_pcd(tempcd)
        # tie points within the bbox
        croppcd = pcd_clean.crop(aabb)
        o3d.io.write_triangle_mesh(os.path.join(args.out, f'{args.prefix}_bbox_p{i}.ply'), cube)
        o3d.io.write_point_cloud(os.path.join(args.out, f'{args.prefix}_p{i}.ply'), croppcd)
