import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ysutils.util_mvsnet import load_cam,read_pfm
import glob
def hypo2depth(hypo,intr_mat):
    h, w, _ = hypo.shape
    h_idx = np.arange(h, dtype=np.int32)
    w_idx = np.arange(w, dtype=np.int32)
    raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
    raw_grid.append(np.ones([h, w]))
    dep_map = -1 * hypo[:, :, -1] * intr_mat[0, 0] / (
            (raw_grid[1] - intr_mat[0, 2]) * hypo[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
            raw_grid[0] - intr_mat[1, 2]) * hypo[:, :, 1] + intr_mat[0, 0] * hypo[:, :, 2])
    return dep_map

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def draw(x):
    plt.imshow(x)
    plt.show()

def draw_hypo(x,vis=False, maxdist = None):
    norm = x[:,:,:3]*0.5 + 0.5
    norm_n = np.stack((norm[:,:,1], norm[:,:,2], norm[:,:,0]), axis=-1)
    norm_n = (norm_n*255).astype("uint8")
    norm_n = adjust_gamma(norm_n, gamma=1.1)
    norm_n = adjust_brightness_contrast(norm_n,1.3,0.5)
    if vis:
        plt.imshow(norm_n)
        plt.show()
    dist = x[:,:,-1]
    if maxdist is None:
        maxdist = dist.max() * 0.8
    dist = (dist/(maxdist) * 255).clip(0,255).astype("uint8")
    colored_dist = cv2.applyColorMap(dist, cv2.COLORMAP_RAINBOW) # bone OCEAN
    colored_dist = cv2.cvtColor(colored_dist, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(colored_dist)
        plt.show()
    return (norm_n,colored_dist)

def draw_cost(cost,vis = False):
    cost_n = cost
    cost_n[cost_n >0.5] = (cost_n[cost_n >0.5] - 0.5) / 1.5 * 0.5 + 0.5
    cost_n = (cost_n * 255).astype("uint8")
    cost_n = adjust_gamma(cost_n, gamma=1.5)
    colored = cv2.applyColorMap(cost_n, cv2.COLORMAP_HOT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

def draw_cost_clip(cost,vis = False,maxcost=None):
    cost_n = cost
    if maxcost is None:
        cost_n[cost_n >0.5] = (cost_n[cost_n >0.5] - 0.5) / 1.5 * 0.5 + 0.5
    else:
        cost_n = (cost_n/maxcost).clip(0,1)
    cost_n = (cost_n * 255).astype("uint8")
    cost_n = adjust_gamma(cost_n, gamma=1.5)
    colored = cv2.applyColorMap(cost_n, cv2.COLORMAP_HOT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

def draw_pred(pred,vis=False):
    pred = (pred * 255).astype("uint8")
    colored = cv2.applyColorMap(pred, cv2.COLORMAP_HOT)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

def draw_depth_hypo(hypo,intrin,vis = False):
    depmap = hypo2depth(hypo,intrin)
    depmap = 255 - ((depmap - depmap.min()) / (depmap.max() - depmap.min()) * 255).astype("uint8")
    colored = cv2.applyColorMap(depmap, cv2.COLORMAP_MAGMA)  # bone OCEAN
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

def draw_depth(depmap,vis = False):
    depmap = 255 - ((depmap - depmap.min()) / (depmap.max() - depmap.min()) * 255).astype("uint8")
    colored = cv2.applyColorMap(depmap, cv2.COLORMAP_MAGMA)  # bone OCEAN
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored


if __name__ == '__main__':
    root = '/mnt/data4/yswangdata4/experiments/optimal'
    caselist = glob.glob(os.path.join(root, '5*'))
    caselist = caselist +  glob.glob(os.path.join(root, 'scan*'))

    saveroot = '/mnt/data4/yswangdata4/experiments/optimal/save'
    os.makedirs(saveroot, exist_ok=True)
    caselist.sort()
    caseid = 3
    id = 100
    cases = [(0,1,250) ,(1,100,20), (3,100,100), (4,1,100), (7,20,150) ,(9,11,150) ,(24,26,2),(26,40,2)]

    if False:
        caseid = 23
        id = 1
        maxdist = 2
        cost_init = np.load(os.path.join(caselist[caseid], 'helix', 'helix_init', f'cost_{id.__str__().zfill(8)}.npy'))
        hypo_init = np.load(os.path.join(caselist[caseid], 'helix', 'helix_init', f'hypo_{id.__str__().zfill(8)}.npy'))
        cost_last = np.load(os.path.join(caselist[caseid], 'helix', 'helix_last', f'cost_{id.__str__().zfill(8)}.npy'))
        hypo_last = np.load(os.path.join(caselist[caseid], 'helix', 'helix_last', f'hypo_{id.__str__().zfill(8)}.npy'))
        diff = np.load(os.path.join(caselist[caseid], 'helix', 'helix_replace', f'replace_{id.__str__().zfill(8)}.npy'))

        c_hypo_init = draw_hypo(hypo_init, vis=True, maxdist=maxdist)
        c_hypo_last = draw_hypo(hypo_last, vis=True, maxdist=maxdist)



    for case in cases:
        caseid = case[0]
        id = case[1]
        maxdist = case[2]
        cost_init = np.load(os.path.join(caselist[caseid],'helix','helix_init',f'cost_{id.__str__().zfill(8)}.npy'))
        hypo_init = np.load(os.path.join(caselist[caseid],'helix','helix_init',f'hypo_{id.__str__().zfill(8)}.npy'))
        cost_last = np.load(os.path.join(caselist[caseid],'helix','helix_last',f'cost_{id.__str__().zfill(8)}.npy'))
        hypo_last = np.load(os.path.join(caselist[caseid],'helix','helix_last',f'hypo_{id.__str__().zfill(8)}.npy'))
        diff = np.load(os.path.join(caselist[caseid],'helix','helix_replace',f'replace_{id.__str__().zfill(8)}.npy'))

        c_hypo_init = draw_hypo(hypo_init,vis=False,maxdist=maxdist)
        c_hypo_last = draw_hypo(hypo_last,vis=False,maxdist=maxdist)
        c_cost_init = draw_cost_clip(cost_init,vis=False,maxcost=0.7)
        c_cost_last = draw_cost_clip(cost_last,vis=False,maxcost=0.7)

        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_init_norm.png'),cv2.cvtColor(c_hypo_init[0], cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_init_dist.png'),cv2.cvtColor(c_hypo_init[1], cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_last_norm.png'),cv2.cvtColor(c_hypo_last[0], cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_last_dist.png'),cv2.cvtColor(c_hypo_last[1], cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_diff.png'),cv2.cvtColor(draw_pred(diff,vis=False), cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_init_cost.png'),cv2.cvtColor(c_cost_init, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(saveroot,f'{caseid}_{id}_last_cost.png'),cv2.cvtColor(c_cost_last, cv2.COLOR_BGR2RGB))




