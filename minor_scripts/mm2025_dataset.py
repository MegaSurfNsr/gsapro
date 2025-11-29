import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ysutils.util_mvsnet import load_cam,read_pfm
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

def draw_hypo(x,vis=False):
    norm = x[:,:,:3]*0.5 + 0.5
    norm_n = np.stack((norm[:,:,1], norm[:,:,2], norm[:,:,0]), axis=-1)
    norm_n = (norm_n*255).astype("uint8")
    norm_n = adjust_gamma(norm_n, gamma=1.1)
    norm_n = adjust_brightness_contrast(norm_n,1.3,0.5)
    if vis:
        plt.imshow(norm_n)
        plt.show()
    dist = x[:,:,-1]
    dist = (dist/dist.max() * 255).astype("uint8")
    colored_dist = cv2.applyColorMap(dist, cv2.COLORMAP_JET) # bone OCEAN
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

    id = [24,72,162,163,247]
    root = '/mnt/data4/yswangdata4/experiments/overview_5af02e904c8216544b4ab5a2/helix_out'
    dataset = '/mnt/data3/yswang2024_data3/dataset/blended_downsample/5af02e904c8216544b4ab5a2'
    save_root = '/mnt/data4/yswangdata4/experiments/overview_5af02e904c8216544b4ab5a2/pic_dataset'
    os.makedirs(save_root, exist_ok=True)
    blended_depth_root = "/mnt/data3/yswang2024_data3/dataset/dataset_low_res/5af02e904c8216544b4ab5a2" #/rendered_depth_maps
    blendedpm_depth_root =  "/mnt/data4/yswangdata4/dataset/dataset_low_res_patchmatch/5af02e904c8216544b4ab5a2/patchmatch" # /patchmatch
    acc_pthred = 0.005
    # target_label = torch.abs(sample_cuda['depth_gt'] - sample_cuda['depth_pm']) > acc_pthred

    cams = [load_cam(os.path.join(dataset,'cams',f'{i.__str__().zfill(8)}_cam.txt')) for i in id]

    # init patchmatch
    costs = []
    costs_i3 = []
    hypos = []
    hypos_i3 = []
    depths = []
    depths_i3 = []
    gts = []

    for i in id:
        costs.append(np.load(os.path.join(blendedpm_depth_root,f'cost_{i.__str__().zfill(8)}.npy')))
        costs_i3.append(np.load(os.path.join(blendedpm_depth_root,f'cost_{i.__str__().zfill(8)}_i3.npy')))
        hypos.append(np.load(os.path.join(blendedpm_depth_root,f'hypo_{i.__str__().zfill(8)}.npy')))
        hypos_i3.append(np.load(os.path.join(blendedpm_depth_root,f'hypo_{i.__str__().zfill(8)}_i3.npy')))
        depths.append(np.load(os.path.join(blendedpm_depth_root,f'depth_{i.__str__().zfill(8)}.npy')))
        depths_i3.append(np.load(os.path.join(blendedpm_depth_root,f'depth_{i.__str__().zfill(8)}_i3.npy')))
        gts.append(read_pfm(os.path.join(blended_depth_root,'rendered_depth_maps',f'{i.__str__().zfill(8)}.pfm'))[0])

    # display
    for i in range(len(id)):
        init_cost = draw_cost(costs[i])
        init_cost_i3 = draw_cost(costs_i3[i])

        init_hypo_norm,init_hypo_dist = draw_hypo(hypos[i])
        init_hypo_norm_i3,init_hypo_dist_i3 = draw_hypo(hypos_i3[i])

        init_hypodepth = draw_depth(depths[i])
        init_hypodepth_i3 = draw_depth(depths_i3[i])

        init_gts = draw_depth(gts[i])

        gt_max = gts[i].max()
        label = (np.abs(gts[i]/gt_max - depths[i]/gt_max) > acc_pthred) * (costs[i] < 0.35)[:,:,0]
        label_i3 = (np.abs(gts[i]/gt_max - depths_i3[i]/gt_max)  > acc_pthred) * (costs_i3[i] < 0.35)[:,:,0]

        label_n = draw_pred(label)
        label_n_i3 = draw_pred(label_i3)

        cv2.imwrite(os.path.join(save_root,f'init_cost_{id[i]}.png'),cv2.cvtColor(init_cost,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_cost_i3_{id[i]}.png'),cv2.cvtColor(init_cost_i3,cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(save_root,f'init_hypo_norm_{id[i]}.png'),cv2.cvtColor(init_hypo_norm,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_hypo_norm_i3_{id[i]}.png'),cv2.cvtColor(init_hypo_norm_i3,cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(save_root,f'init_hypo_dist_{id[i]}.png'),cv2.cvtColor(init_hypo_dist,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_hypo_dist_i3_{id[i]}.png'),cv2.cvtColor(init_hypo_dist_i3,cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(save_root,f'init_hypodepth_{id[i]}.png'),cv2.cvtColor(init_hypodepth,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_hypodepth_i3_{id[i]}.png'),cv2.cvtColor(init_hypodepth_i3,cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(save_root,f'init_gt_{id[i]}.png'),cv2.cvtColor(init_gts,cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(save_root,f'init_gtlabel_{id[i]}.png'),cv2.cvtColor(label_n,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_gtlabel_i3_{id[i]}.png'),cv2.cvtColor(label_n_i3,cv2.COLOR_BGR2RGB))





    print('test')