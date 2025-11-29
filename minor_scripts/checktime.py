import tqdm
import glob
import os
import numpy as np
import open3d as o3d
import time

def get_file_creation_time(file_path):
    try:
        file_stat = os.stat(file_path)
        if os.name == 'nt':
            creation_time = file_stat.st_ctime
        else:
            creation_time = file_stat.st_ctime
        # creation_time_str = time.ctime(creation_time)
        return creation_time
    except FileNotFoundError:
        print(f" {file_path} not found")
    except Exception as e:
        print(f"err {e}")

if __name__ == '__main__':
    root1 = glob.glob('/mnt/data4/yswangdata4/experiments/dtu_gs2d/gs2d_scan*/input.ply')
    root2 = glob.glob('/mnt/data4/yswangdata4/experiments/dtu_gs2d/gs2d_scan*/point_cloud/iteration_30000/point_cloud.ply')

    root1.sort()
    root2.sort()
    assert len(root1) == len(root2)
    times1 = [get_file_creation_time(f) for f in root1]
    times2 = [get_file_creation_time(f) for f in root2]
    dif = [(f2-f1)/60 for f1,f2 in zip(times1,times2)]
    dif.sort()


    root1 = glob.glob('/mnt/data3/wangan/zh/ori_code/gaussian_surfels/old_output/scan*/cfg_args')
    root2 = glob.glob('/mnt/data3/wangan/zh/ori_code/gaussian_surfels/old_output/scan*/point_cloud/iteration_15000/point_cloud.ply')

    root1.sort()
    root2.sort()
    assert len(root1) == len(root2)
    times1 = [get_file_creation_time(f) for f in root1]
    times2 = [get_file_creation_time(f) for f in root2]
    dif = [(f2-f1)/60 for f1,f2 in zip(times1,times2)]
    dif.sort()
    print(dif)





