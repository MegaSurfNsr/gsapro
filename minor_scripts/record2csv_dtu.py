import glob
from ysutils.util_mvsnet import read_pfm
from matplotlib import pyplot as plt
import numpy as np
import os


if __name__ == '__main__':
    cases_root = '/mnt/data4/yswangdata4/experiments/multiview_filter/dtu'
    cases = glob.glob(os.path.join(cases_root, '*scan*','*.csv'))
    cases.sort()

    print('pause test')
    # read rec
    recdict = {}
    for case in cases:
        with open(case,'r') as f:
            rec = f.readlines()[-1].strip().split(',')
        recdict[case.split('/')[-2]] = rec

    # write summary
    # '/mnt/data4/yswangdata4/experiments/mm25evaluation/mm2025_evaluation.txt'
    with open('/mnt/data4/yswangdata4/experiments/mm25evaluation/mm2025_evaluation_dtu.txt','a') as f:
        for scene in recdict.keys():
            f.write(scene)
            for r in recdict[scene]:
                f.write(',' + r)
            f.write('\n')
