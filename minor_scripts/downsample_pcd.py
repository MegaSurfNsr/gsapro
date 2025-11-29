'/mnt/data3/yswang2024_data3/dataset/blended_highres/scene'
'/mnt/data3/yswang2024_data3/dataset/blended_mesh'
import open3d as o3d
import os
import glob
import tqdm
for pcd_file in tqdm.tqdm(glob.glob('/mnt/data3/yswang2024_data3/dataset/blended_mesh/*.ply')):
    if os.path.exists(os.path.join(os.path.dirname(pcd_file), f'down_{os.path.basename(pcd_file)}')):
        continue
    if os.path.basename(pcd_file)[:4]=='down':
        continue
    pcd = o3d.io.read_point_cloud(pcd_file)
    ratio = min(6_000_000 / len(pcd.points), 1)
    dpcd = pcd.random_down_sample(ratio)
    dpcd = dpcd.voxel_down_sample(voxel_size=0.025)
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(pcd_file), f'down_{os.path.basename(pcd_file)}'),dpcd)


for pcd_file in tqdm.tqdm(glob.glob('/mnt/data3/yswang2024_data3/helix_output/partsave/*.ply')):
    pcd = o3d.io.read_point_cloud(pcd_file)
    ratio = min(6_000_000 / len(pcd.points), 1)
    print(pcd_file)
    print(ratio)
    if ratio >= 0.9999:
        continue
    print('downsampling')
    dpcd = pcd.random_down_sample(ratio)
    # dpcd = dpcd.voxel_down_sample(voxel_size=0.025)
    o3d.io.write_point_cloud(pcd_file,dpcd)
