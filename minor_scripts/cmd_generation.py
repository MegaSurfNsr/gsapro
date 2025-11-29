import os
import glob
import math

gpus = 4
gpuidx = [6,6,7,7]
'/mnt/data3/yswang2024_data3/helix_output/helixonlyncc_5b7a3890fc8fcf6781e2593a' # lresult path
'/mnt/data3/yswang2024_data3/helix_output/partsave' # initply path
'/mnt/data3/yswang2024_data3/helixpart_output' # outpath

cmds = []
with open('/mnt/data3/yswang2024_data3/dataset/blended_highres/process_scene', 'r') as f:
    scenelist = f.readlines()
    scenelist = [s.strip() for s in scenelist]

for scene in scenelist:
    partlist = glob.glob(os.path.join('/mnt/data3/yswang2024_data3/helix_output/partsave',f'{scene}*.ply'))
    for part in partlist:
        partname = os.path.basename(part).split('.')[0]
        outpath = os.path.join('/mnt/data3/yswang2024_data3/helixpart_output',partname)
        lresult = os.path.join(f'/mnt/data3/yswang2024_data3/helix_output/helixonlyncc_{scene}')
        initply = part
        cmd = f'python train_part.py -s /mnt/data3/yswang2024_data3/dataset/blended_downsample/{scene} -m {outpath} --data_device cpu --densify_abs_grad_threshold 0.0004 --lresult {lresult} --initply {initply}'
        cmds.append(cmd)

n_perf = math.ceil(len(cmds) / gpus)
l = 0
for i in range(gpus):
    with open(f'/home/yswang/data3/code/PGSR_main/commands/cmd_{i}.sh', 'w') as f:
        for j in range(n_perf):
            if l >= len(cmds):
                break
            cmd = cmds[l]
            cmd = f'CUDA_VISIBLE_DEVICES={gpuidx[i]} ' + cmd + '\n'
            f.write(cmd)
            l = l + 1



# cmd = f'python train_part.py -s /mnt/data3/yswang2024_data3/dataset/blended_downsample/{scene} -m {outpath} --data_device cpu --densify_abs_grad_threshold 0.0004 --lresult {lresult} --initply {initply}'