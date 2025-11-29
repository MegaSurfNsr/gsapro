import os


if __name__ == '__main__':

    scenes = ['Barn', 'Courthouse', 'Truck', 'Caterpillar', 'Meetingroom', 'Ignatius']
    data_devices = ['cpu', 'cpu', 'cpu', 'cpu', 'cpu', 'cpu']
    data_base_path = '/mnt/data4/yswangdata4/dataset/tnt_train'

    gpu_id = 6

    out_base_path = '/mnt/data4/yswangdata4/experiments/tnt_out'
    out_name = 'train_helix_ncc_from_pgsr_3w'
    for id, scene in enumerate(scenes):
        # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt_new.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]}'
        # print(cmd)
        # os.system(cmd)
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt_new.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
        print(cmd)
        try:
            os.system(cmd)
        except:
            pass

    out_base_path = '/mnt/data4/yswangdata4/experiments/pgsr_tnt'
    out_name = 'test'
    for id, scene in enumerate(scenes):
        # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt_new.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]}'
        # print(cmd)
        # os.system(cmd)
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt_new.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
        print(cmd)
        try:
            os.system(cmd)
        except:
            pass

    out_base_path = '/mnt/data4/yswangdata4/experiments/tnt_out'
    out_name = 'train_helix_from_pgsr_3w'
    for id, scene in enumerate(scenes):
        # cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt_new.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]}'
        # print(cmd)
        # os.system(cmd)
        common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt_new.py -m {out_base_path}/{scene}/{out_name} --data_device {data_devices[id]} {common_args}'
        print(cmd)
        try:
            os.system(cmd)
        except:
            pass


    # for id, scene in enumerate(scenes):
    #     # require open3d==0.9
    #     cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{scene}/{out_name}/mesh/tsdf_fusion.ply --out-dir {out_base_path}/{scene}/{out_name}/mesh'
    #     print(cmd)
    #     os.system(cmd)