import os
import shutil
roots = ['/mnt/data4/yswangdata4/experiments/pg2025exp/mvsformer_blend','/mnt/data4/yswangdata4/experiments/pg2025exp/mvsformerpp_blend_test','/mnt/data4/yswangdata4/experiments/pg2025exp/ours_blended_noise']
lists = '/mnt/data4/yswangdata4/code/mvsformerpp/lists/blended/old_validation_list.txt'

with open(lists, 'r') as f:
    scenelist = [s.strip() for s in f.readlines()]
for root in roots:
    files = os.listdir(root)
    for f in files:
        if '\r' in f:
            shutil.move(os.path.join(root, f), os.path.join(root, f.replace('\r','')))
            subroot = os.path.join(root, f.replace('\r',''))
            subfiles = os.listdir(subroot)
            for subf in subfiles:
                if '\r' in subf:
                    shutil.move(os.path.join(root, subroot,subf), os.path.join(root, subroot,subf).replace('\r',''))
