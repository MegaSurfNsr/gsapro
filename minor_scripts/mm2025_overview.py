import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ysutils.util_mvsnet import load_cam

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
    colored = cv2.applyColorMap(depmap, cv2.COLORMAP_TWILIGHT_SHIFTED)  # bone OCEAN
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored

def draw_depth(depmap,vis = False):
    depmap = 255 - ((depmap - depmap.min()) / (depmap.max() - depmap.min()) * 255).astype("uint8")
    colored = cv2.applyColorMap(depmap, cv2.COLORMAP_TWILIGHT_SHIFTED)  # bone OCEAN
    if vis:
        plt.imshow(colored)
        plt.show()
    return colored


if __name__ == '__main__':
    id = [24,72,162,163,247]
    root = '/mnt/data4/yswangdata4/experiments/overview_5af02e904c8216544b4ab5a2/helix_out'
    dataset = '/mnt/data3/yswang2024_data3/dataset/blended_downsample/5af02e904c8216544b4ab5a2'
    save_root = '/mnt/data4/yswangdata4/experiments/overview_5af02e904c8216544b4ab5a2/pic'
    os.makedirs(save_root, exist_ok=True)

    # blended_depth_root = "/mnt/data3/yswang2024_data3/dataset/dataset_low_res/5af02e904c8216544b4ab5a2" #/rendered_depth_maps
    # blendedpm_depth_root =  "/mnt/data4/yswangdata4/dataset/dataset_low_res_patchmatch/5af02e904c8216544b4ab5a2" # /patchmatch
    # acc_pthred = 0.35

    # target_label = torch.abs(sample_cuda['depth_gt'] - sample_cuda['depth_pm']) > acc_pthred


    cams = [load_cam(os.path.join(dataset,'cams',f'{i.__str__().zfill(8)}_cam.txt')) for i in id]

    # init patchmatch
    costs = []
    hypos = []
    depths = []
    sems = []
    sempreds = []
    renderhypos = []
    for i in id:
        costs.append(np.load(os.path.join(root,'helix',f'cost_{i.__str__().zfill(8)}.npy')))
        hypos.append(np.load(os.path.join(root,'helix',f'hypo_{i.__str__().zfill(8)}.npy')))
        depths.append(np.load(os.path.join(root,'helix_it0',f'depth_{i.__str__().zfill(8)}.npy')))
        sems.append(np.load(os.path.join(root,'helix_it0',f'seg_{i.__str__().zfill(8)}.npy')))
        sempreds.append(np.load(os.path.join(root,'helix_it0',f'segpred_{i.__str__().zfill(8)}.npy')))
        renderhypos.append(np.load(os.path.join(root,'pgsr_it3000',f'hypo_{i.__str__().zfill(8)}.npy')))

    # better results
    bcosts = []
    bhypos = []
    bdepths = []
    bsems = []
    bsempreds = []
    brenderhypos = []
    for i in id:
        bcosts.append(np.load(os.path.join(root,'helix_it15000',f'cost_{i.__str__().zfill(8)}.npy')))
        bhypos.append(np.load(os.path.join(root,'helix_it15000',f'hypo_{i.__str__().zfill(8)}.npy')))
        bsems.append(np.load(os.path.join(root,'helix_it15000',f'seg_{i.__str__().zfill(8)}.npy')))
        bsempreds.append(np.load(os.path.join(root,'helix_it15000',f'segpred_{i.__str__().zfill(8)}.npy')))
        bdepths.append(np.load(os.path.join(root,'helix_it15000',f'depth_{i.__str__().zfill(8)}.npy')))
        brenderhypos.append(np.load(os.path.join(root,'pgsr_it7000',f'hypo_{i.__str__().zfill(8)}.npy')))

    # display
    for i in range(len(id)):
        init_cost = draw_cost(costs[i])
        init_hypo_norm,init_hypo_dist = draw_hypo(hypos[i])
        init_hypodepth = draw_depth_hypo(hypos[i],cams[i][1][:3,:3])
        init_guidedepth = draw_depth(depths[i])
        init_pred = draw_pred(sempreds[i])
        init_render_norm,init_render_dist = draw_hypo(renderhypos[i])

        cv2.imwrite(os.path.join(save_root,f'init_cost_{id[i]}.png'),cv2.cvtColor(init_cost,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_hypo_norm_{id[i]}.png'),cv2.cvtColor(init_hypo_norm,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_hypo_dist_{id[i]}.png'),cv2.cvtColor(init_hypo_dist,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_hypodepth_{id[i]}.png'),cv2.cvtColor(init_hypodepth,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_guidedepth_{id[i]}.png'),cv2.cvtColor(init_guidedepth,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_render_norm_{id[i]}.png'),cv2.cvtColor(init_render_norm,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_render_dist_{id[i]}.png'),cv2.cvtColor(init_render_dist,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'init_pred_{id[i]}.png'),cv2.cvtColor(init_pred,cv2.COLOR_BGR2RGB))


        b_cost = draw_cost(bcosts[i])
        b_hypo_norm,b_hypo_dist = draw_hypo(bhypos[i])
        b_hypodepth = draw_depth_hypo(bhypos[i],cams[i][1][:3,:3])
        b_guidedepth = draw_depth(bdepths[i])
        b_pred = draw_pred(bsempreds[i])
        b_render_norm,b_render_dist = draw_hypo(brenderhypos[i])

        cv2.imwrite(os.path.join(save_root,f'b_cost_{id[i]}.png'),cv2.cvtColor(b_cost,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_hypo_norm_{id[i]}.png'),cv2.cvtColor(b_hypo_norm,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_hypo_dist_{id[i]}.png'),cv2.cvtColor(b_hypo_dist,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_hypodepth_{id[i]}.png'),cv2.cvtColor(b_hypodepth,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_guidedepth_{id[i]}.png'),cv2.cvtColor(b_guidedepth,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_pred_{id[i]}.png'),cv2.cvtColor(b_pred,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_render_norm_{id[i]}.png'),cv2.cvtColor(b_render_norm,cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(save_root,f'b_render_dist_{id[i]}.png'),cv2.cvtColor(b_render_dist,cv2.COLOR_BGR2RGB))






    print('test')