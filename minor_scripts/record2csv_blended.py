import os
import numpy as np

if __name__ == '__main__':

    print('pause test')
    # read rec
    with open('/mnt/data4/yswangdata4/experiments/pg2025exp/mvsformerpp_blend_test/evaluation_downvoxel.txt','r') as f:
        rec = f.readlines()

    recdict = {}
    method_prefix = 'mvsformerpp'
    i = 0
    while i < len(rec):
        name = rec[i].split('/')[-3]
        method = name
        scene = rec[i].split('/')[-2]
        if method_prefix is not None:
            method = method_prefix
        if scene not in recdict.keys():
            recdict[scene] = {}
        recdict[scene][method] = rec[i+3].strip().strip('(').strip(')')
        i = i + 4

    # write summary
    # '/mnt/data4/yswangdata4/experiments/mm25evaluation/mm2025_evaluation.txt'
    with open('/mnt/data4/yswangdata4/experiments/pg2025exp/pg2025_blended.txt','a') as f:
        for scene in recdict.keys():
            # f.write(scene+'\n')
            for method in recdict[scene].keys():
                f.write(scene+', ' + method+', ' + recdict[scene][method]+'\n')

    # calculate final evaluation score



    # import matplotlib.pyplot as plt
    # import numpy as np
    # elimit = np.load('/mnt/data3/yswang2024_data3/helix_output/helix_v2_5ba75d79d76ffa2c86cf2f05/helix_out/dv2mask_it30000/elimit_00000000.npy')