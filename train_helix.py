#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import glob
import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui, render_imp
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
from helix_model.helixncc import HelixNCC,transform_normalmap
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ysutils.util_ncc import calncc_patch
from utils.sh_utils import SH2RGB
import shutil
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def region_minmax(map,npflag=True):
    with torch.no_grad():
        dev_map = torch.tensor(map).cuda().unsqueeze(0).unsqueeze(0)
        kernel_size = (15,15)
        maxpool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size[0]//2)
        map_max = maxpool(dev_map)

        dev_map2 = 1 - dev_map
        dev_map2[dev_map2 >0.99] = 0
        map_min = 1 - maxpool(dev_map2)

        if npflag:
            map_max = map_max.cpu().squeeze(0).squeeze(0).numpy()
            map_min = map_min.cpu().squeeze(0).squeeze(0).numpy()
    return map_min, map_max

def classify(class1,class2,input):
    dif1 = np.abs(class1-input)
    dif2 = np.abs(class2-input)
    return dif1 < dif2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(22)


def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                        preload_img=False, data_device="cuda")
    return virtul_cam


def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()
    if thres != 1.0:
        percent_sum = thres
        vals, idx = torch.sort(importance + (1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val / vals.sum()) > (1 - percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance > split_val_nonprune
    else:
        non_prune_mask = torch.ones_like(importance).bool()

    return non_prune_mask



def generate_plane_hypo(depth,normal,K):
    h,w= depth.shape
    h_idx = np.arange(h, dtype=np.int32)
    w_idx = np.arange(w, dtype=np.int32)
    raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
    raw_grid.append(np.ones([h, w]))
    x = depth * (raw_grid[1] - K[0,2]) / K[0,0]
    y = depth * (raw_grid[0] - K[1, 2]) / K[1, 1]
    z = depth
    w = -1 * (normal[:, :, 0] * x + normal[:, :, 1] * y + normal[:, :,2] * z)
    return np.concatenate([normal,w[:,:,None]], axis=-1)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             helixncc=None,helix_iterations=[30001],miniprune=True,segmodel=None):
    print(f'miniprune: {miniprune} ')
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code

    runfile = sys.argv[0]
    os.makedirs(dataset.model_path + '/code', exist_ok=True)
    shutil.copyfile(runfile, os.path.join(dataset.model_path, 'code','run.py'))
    shutil.copytree(os.path.join(os.path.dirname(runfile),'models'), os.path.join(dataset.model_path, 'code','models'),dirs_exist_ok=True)
    shutil.copytree(os.path.join(os.path.dirname(runfile),'arguments'), os.path.join(dataset.model_path, 'code','arguments'),dirs_exist_ok=True)

    # shutil.copytree(os.path.join(os.path.dirname(runfile),'dataset'), os.path.join(dataset.model_path, 'code','dataset'),dirs_exist_ok=True)
    # shutil.copytree(os.path.join(os.path.dirname(runfile),'config'), os.path.join(dataset.model_path, 'code','config'),dirs_exist_ok=True)



    # gaussians = GaussianModel(dataset.sh_degree)
    if miniprune:
        gaussians = GaussianModel(sh_degree=0)
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # perform the semantic process
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack.sort(key=lambda x: x.image_name)

    flag_tnt = False
    if int(viewpoint_stack[-1].image_name) == len(viewpoint_stack):
        flag_tnt=True

    print('semantic mask generation')
    for i in tqdm(range(len(viewpoint_stack))):
        process_cam = viewpoint_stack[i]
        # helix_id = int(process_cam.image_name)
        helix_id = scene.imglist.index(process_cam.image_name)

        img = process_cam.original_image.cuda().unsqueeze(0)
        helix_depth = helixncc.depths[helix_id].cuda()
        helix_depth = helix_depth / (helix_depth.max() /1.05)
        helix_depth = helix_depth.unsqueeze(0)
        helix_cost = helixncc.costs[helix_id].cuda().unsqueeze(0)
        with torch.no_grad():
            b,c,h,w = img.shape
            h_inter = h//8 * 8
            w_inter = w//8 * 8
            if h_inter != h:
                img_seg = F.interpolate(img, (h_inter, w_inter), mode='nearest')
                helix_depth_seg = F.interpolate(helix_depth.float(), (h_inter, w_inter), mode='nearest')
                helix_cost_seg = F.interpolate(helix_cost, (h_inter, w_inter), mode='nearest')
                out = segmodel(img_seg,helix_depth_seg.float() * (helix_cost_seg < helixncc.meta['ncc_pthred']),helix_cost_seg)
                out[-1] = F.interpolate(out[-1], (h, w), mode='nearest')
            else:
                out = segmodel(img,helix_depth.float() * (helix_cost < helixncc.meta['ncc_pthred']),helix_cost)
        helixncc.seg_mask.append(out[-1][0].cpu())
        fmask = ((helix_cost < helixncc.meta['ncc_pthred']) * (out[-1][0] < 0.5)).cpu()[0]
        helixncc.depths[helix_id][torch.logical_not(fmask)] = -1

        if helixncc.meta['savehelix']:
            if i % 20 == 0:
                os.makedirs(os.path.join(helixncc.meta['savepath'], f'helix_it{0}'), exist_ok=True)
                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{0}', f'depth_{i.__str__().zfill(8)}'),
                        helixncc.depths[helix_id][0].numpy())
                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{0}', f'cost_{i.__str__().zfill(8)}'),
                        helixncc.costs[helix_id][0].numpy())
                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{0}', f'seg_{i.__str__().zfill(8)}'),
                        fmask[0].numpy())
                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{0}', f'segpred_{i.__str__().zfill(8)}'),
                        out[-1][0].cpu()[0].numpy())

        # plt.imshow(out[-1].cpu()[0, 0])
        # plt.show()
        # plt.imshow(helix_depth.cpu()[0,0])
        # plt.show()
        # plt.imshow(helix_cost.cpu()[0,0]<0.20)
        # plt.show()
    #



    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="porg")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # search id for helix
        helix_id = scene.imglist.index(viewpoint_cam.image_name)

        helix_depth = helixncc.depths[helix_id].cuda()

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=True,
                            return_depth_normal=iteration > opt.single_view_weight_from_iter)
        # return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)

        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        pgsr_depth = render_pkg['plane_depth']
        # render_pkg['plane_depth'][0]

        # Loss
        helix_mask = helix_depth > 0.001
        helix_dif = torch.tanh(helix_depth - pgsr_depth).abs()
        helix_depth_loss = helix_dif[helix_mask].mean() #*0.25

        # helix_mask_edge = torch.logical_and(helix_depth < 0 , helix_depth > -0.5)
        # helix_depth_edge_loss = helix_dif[helix_mask_edge].mean() * 0.5

        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()

        loss += helix_depth_loss #* 0.5  #* 0.25
        if TENSORBOARD_FOUND:
            tb_writer.add_scalar('train_loss_patches/helix_depth_loss', helix_depth_loss.item(), iteration)

        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[..., 0]
            loss += 100.0 * min_scale_loss.mean()

        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0, 1).detach() ** 5
            image_weight = erode(image_weight[None, None]).squeeze()

            normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            loss += (normal_loss)

            # normal_grad_x = (normal[:,1:-1,:-2] - normal[:,1:-1,2:]).abs()
            # normal_grad_y = (normal[:,:-2,1:-1] - normal[:,2:,1:-1]).abs()
            # smooth_normal_loss = 0.002 * ((normal_grad_x + normal_grad_y)).mean()
            # distance_grad_x = (render_pkg['rendered_distance'][:,1:-1,:-2] - render_pkg['rendered_distance'][:,1:-1,2:]).abs()
            # distance_grad_y = (render_pkg['rendered_distance'][:,:-2,1:-1] - render_pkg['rendered_distance'][:,2:,1:-1]).abs()
            # smooth_distance_loss = 0.002 * (image_weight[None,1:-1,1:-1] * (distance_grad_x + distance_grad_y)).mean()
            # loss += (smooth_normal_loss)
            # loss += (smooth_distance_loss)

        # multi-view loss
        opt.multi_view_weight_from_iter = 300010
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[
                random.sample(viewpoint_cam.nearest_id, 1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis,
                                             deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[
                                                                                      3, :3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'],
                                                                        pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,
                                         :3] + viewpoint_cam.world_view_transform[3, :3]
                pts_projections = torch.stack(
                    [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                     pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                if iteration % 200 == 0:
                    gt_img_show = ((gt_image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                   [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1, 2, 0).clamp(0, 1)[:, :,
                                    [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                    [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,
                                                                                 1) * 255).detach().cpu().numpy().astype(
                        np.uint8)
                    depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,
                                                                                             1) * 255).detach().cpu().numpy().astype(
                        np.uint8)
                    d_mask_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8).reshape(H, W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show], axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"),
                                image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1, 2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                                         align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1,
                                                                                                   -2) @ viewpoint_cam.world_view_transform[
                                                                                                         :3, :3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,
                                                                     :3] + nearest_cam.world_view_transform[3, :3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0)
                        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = rendered_normal.reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(H,W)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                                            torch.matmul(
                                                ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                                ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2,
                                                                                                                   1)) / \
                                            ref_local_d[..., None, None]
                        H_ref_to_neareast = torch.matmul(
                            nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
                            H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2),
                                                         align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss

        loss.backward()
        iter_end.record()

        # if iteration == 3000:
        #     print('pause')

        # renew prior
        with torch.no_grad():
            if iteration in helix_iterations or iteration == opt.iterations:
                os.makedirs(os.path.join(helixncc.meta['savepath'], f'helix_it{iteration}'),exist_ok=True)
                os.makedirs(os.path.join(helixncc.meta['savepath'], f'pgsr_it{iteration}'),exist_ok=True)

                helixncc.set_pert_ratio(helixncc.helix_ratio*0.5)
                # pgsr render prior
                for idx, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
                    print(view.image_name)
                    helix_id = scene.imglist.index(view.image_name)

                    out = render(view, gaussians, pipe, bg, app_model=app_model)
                    depth = out['plane_depth'].squeeze()
                    depth = depth.detach().cpu().numpy()
                    H, W = depth.shape
                    normal = out["rendered_normal"].permute(1, 2, 0)
                    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)
                    normal = normal.detach().cpu().numpy()
                    K = view.get_k().detach().cpu().numpy()
                    render_hypo = generate_plane_hypo(depth,normal,K)
                    if iteration == opt.iterations or helixncc.meta['savehelix']:
                        np.save(os.path.join(helixncc.meta['savepath'], f'pgsr_it{iteration}', f'hypo_{view.image_name}'), render_hypo)
                    # helixncc.priors[helix_id] = hypo

                    if iteration in helix_iterations:
                        i = helix_id
                        previous_depth = helixncc.depths[i]

                        try:
                            dep_min = previous_depth[previous_depth > 0].min().item()
                            dep_max = previous_depth[previous_depth > 0].max().item()
                            valid_new_mask = (depth > dep_min) * (depth < dep_max)
                        except:
                            valid_new_mask = np.ones_like(depth).astype(np.bool)

                        helixncc.process_with_prior_manual(i,render_hypo)
                        previous_cost = helixncc.costs[i]
                        new_cost = helixncc.cost[:,:,0].cpu().unsqueeze(0)

                        replace_mask = (previous_cost > (new_cost + 0.05)).numpy() * valid_new_mask[None,:,:]
                        helixncc.priors[helix_id][replace_mask[0]] = render_hypo[replace_mask[0]]
                        helixncc.process_with_prior(i)


                        helixout = helixncc.hypos.cpu().numpy()
                        cost = helixncc.cost.cpu().numpy()
                        helixncc.priors[i] = helixout

                        intr_mat = helixncc.cams[i][0][:3, :3]
                        h, w, _ = cost.shape
                        h_idx = np.arange(h, dtype=np.int32)
                        w_idx = np.arange(w, dtype=np.int32)
                        raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
                        raw_grid.append(np.ones([h, w]))
                        dep_map = -1 * helixout[:, :, -1] * intr_mat[0, 0] / (
                                (raw_grid[1] - intr_mat[0, 2]) * helixout[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
                                raw_grid[0] - intr_mat[1, 2]) * helixout[:, :, 1] + intr_mat[0, 0] * helixout[:, :, 2])

                        pthres = helixncc.meta['ncc_pthred']

                        depth = torch.from_numpy(dep_map).unsqueeze(0)
                        cost = torch.from_numpy(cost).unsqueeze(0).squeeze(-1)

                        helixncc.depths[i] = depth
                        helixncc.costs[i] = cost

                        # if helixncc.maskflag:
                        #     dep_map[np.logical_not(helixncc.masks[i])] = -1

                        # seg net process

                        process_cam = view
                        # helix_id = int(process_cam.image_name)
                        helix_id = scene.imglist.index(process_cam.image_name)

                        img = process_cam.original_image.cuda().unsqueeze(0)
                        helix_depth = helixncc.depths[helix_id].cuda()
                        helix_depth = helix_depth / (helix_depth.max() / 1.05)
                        helix_depth = helix_depth.unsqueeze(0)
                        helix_cost = helixncc.costs[helix_id].cuda().unsqueeze(0)

                        with torch.no_grad():
                            b, c, h, w = img.shape
                            h_inter = h // 8 * 8
                            w_inter = w // 8 * 8
                            if h_inter != h:
                                img_seg = F.interpolate(img, (h_inter, w_inter), mode='nearest')
                                helix_depth_seg = F.interpolate(helix_depth.float(), (h_inter, w_inter), mode='nearest')
                                helix_cost_seg = F.interpolate(helix_cost, (h_inter, w_inter), mode='nearest')
                                out = segmodel(img_seg,
                                               helix_depth_seg.float() * (helix_cost_seg < helixncc.meta['ncc_pthred']),
                                               helix_cost_seg)
                                out[-1] = F.interpolate(out[-1], (h, w), mode='nearest')
                            else:
                                out = segmodel(img, helix_depth.float() * (helix_cost < helixncc.meta['ncc_pthred']),
                                               helix_cost)

                        helixncc.seg_mask[helix_id] = out[-1][0].cpu()
                        fmask = ((helix_cost < helixncc.meta['ncc_pthred']) * (out[-1][0] < 0.5)).cpu()[0]
                        helixncc.depths[helix_id][torch.logical_not(fmask)] = -1

                        if helixncc.meta['savehelix']:
                            if i % 20 == 0:
                                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{iteration}', f'hypo_{i.__str__().zfill(8)}'),
                                        helixncc.hypos.cpu().numpy())
                                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{iteration}', f'depth_{i.__str__().zfill(8)}'), helixncc.depths[helix_id][0].numpy())
                                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{iteration}', f'cost_{i.__str__().zfill(8)}'), helixncc.costs[helix_id][0].numpy())
                                np.save(os.path.join(helixncc.meta['savepath'], f'helix_it{iteration}', f'seg_{i.__str__().zfill(8)}'), fmask[0].numpy())
                                np.save(
                                    os.path.join(helixncc.meta['savepath'], f'helix_it{iteration}', f'segpred_{i.__str__().zfill(8)}'),
                                    out[-1][0].cpu()[0].numpy())




        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "Single": f"{ema_single_view_for_log:.{5}f}",
                    # "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    # "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)

            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False,
                                            return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # if (iteration in [2000, 4000, 6000, 8000, 10000, 12000, 14000] or gaussians._xyz.shape[
            #     0] > 4000000) and miniprune:
            # if (iteration in [4000,8000,12000,14000] or gaussians._xyz.shape[0] > 4000000) and miniprune:
            if (iteration in [8000, 14000] or gaussians._xyz.shape[0] > 4000000) and miniprune:

                print('Prune Periodically')
                imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
                accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
                views = scene.getTrainCameras()
                for view in views:
                    gt = view.original_image[0:3, :, :]

                    render_pkg = render_imp(view, gaussians, pipe, background)
                    accum_weights = render_pkg["accum_weights"]
                    area_proj = render_pkg["area_proj"]
                    area_max = render_pkg["area_max"]

                    accum_area_max = accum_area_max + area_max

                    # if args.imp_metric == 'outdoor':
                    mask_t = area_max != 0
                    temp = imp_score + accum_weights / area_proj
                    imp_score[mask_t] = temp[mask_t]
                    # else:
                    #     imp_score = imp_score + accum_weights

                imp_score[accum_area_max == 0] = 0
                non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99)

                gaussians.prune_points(non_prune_mask == False)
                # gaussians.training_setup(opt)
                torch.cuda.empty_cache()
            if iteration == 10000 and miniprune:
                gaussians.max_sh_degree = dataset.sh_degree
                gaussians.reinitial_pts(gaussians._xyz,
                                    SH2RGB(gaussians._features_dc+0)[:,0])
                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                viewpoint_stack = scene.getTrainCameras().copy()
            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

            if iteration % 500 == 0:
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6207)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_010])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,10000, 30000])
    parser.add_argument("--helix_iterations", nargs="+", type=int, default=[3000, 8000,15000,20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # parser.add_argument("--maskdir", type=str, default=None)
    parser.add_argument("--segnet", type=str, default='/mnt/data4/yswangdata4/code/trimnet/expout/blended_0327/model_000035.ckpt')
    parser.add_argument("--save_helix", action='store_true')
    parser.add_argument("--pair", type=str, default=None)
    parser.add_argument("--miniprune", action='store_true')
    parser.add_argument("--onlyncc", action='store_true')
    parser.add_argument('--ncc_pthred', default=0.5, type=float) #0.35
    parser.add_argument('--npair', default=10, type=int)



    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # prepare segnet
    from models.trimnet_for_helix_0328 import DINOv2Trimnet
    seg_model = DINOv2Trimnet(None)
    seg_model.to('cuda')
    loadckpt = os.path.join(args.segnet)
    print("pretrain ckpt ", loadckpt)
    state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
    seg_model.load_state_dict(state_dict['model'])
    seg_model.eval()


    imgpath = os.path.join(args.source_path, 'images')
    savepath = os.path.join(args.model_path, 'helix_out')
    camspath = os.path.join(args.source_path, 'cams')
    if args.pair is not None:
        pairpath = args.pair
    else:
        pairpath = os.path.join(args.source_path, 'pair.txt')




    # os.makedirs(os.path.join(savepath, 'lowres'), exist_ok=True)
    helix_resolution = args.resolution if args.resolution>=1 else 1
    helixncc = HelixNCC(camspath,pairpath,None,imgpath,dscale=helix_resolution,pairneighbor=args.npair)
    helixncc.meta['savepath'] = savepath
    helixncc.meta['savehelix'] = args.save_helix
    helixncc.meta['ncc_pthred'] = args.ncc_pthred

    # if args.maskdir:
    #     maskpaths = glob.glob(os.path.join(args.maskdir, '*'))
    #     maskpaths.sort()
    #     helixncc.meta['maskpaths'] = maskpaths
    #     helixncc.maskflag = True
    #     helixncc.load_mask()


    # init training
    for i in tqdm(range(helixncc.n_image)):
        helixncc.process_pair_data(i)

        os.makedirs(os.path.join(savepath,'helix'),exist_ok=True)
        helixout = helixncc.hypos.cpu().numpy()
        helixncc.priors.append(helixout)
        cost = helixncc.cost.cpu().numpy()
        intr_mat = helixncc.cams[i][0][:3,:3]
        h, w, _ = cost.shape
        h_idx = np.arange(h, dtype=np.int32)
        w_idx = np.arange(w, dtype=np.int32)
        raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
        raw_grid.append(np.ones([h, w]))
        dep_map = -1 * helixout[:, :, -1] * intr_mat[0, 0] / (
                (raw_grid[1] - intr_mat[0, 2]) * helixout[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
                raw_grid[0] - intr_mat[1, 2]) * helixout[:, :, 1] + intr_mat[0, 0] * helixout[:, :, 2])
        pthres = args.ncc_pthred

        depth = torch.from_numpy(dep_map).unsqueeze(0)
        cost = torch.from_numpy(cost).unsqueeze(0).squeeze(-1)

        if args.save_helix:
            if i % 20 == 0:
                np.save(os.path.join(savepath,'helix',f'hypo_{i.__str__().zfill(8)}'),helixncc.hypos.cpu())
                np.save(os.path.join(savepath,'helix',f'cost_{i.__str__().zfill(8)}'),helixncc.cost.cpu())

        helixncc.depths.append(depth)
        helixncc.costs.append(cost)

    if args.onlyncc:
        exit()
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, helixncc=helixncc, helix_iterations=args.helix_iterations,miniprune=args.miniprune,segmodel=seg_model)

    # All done
    print("\nTraining complete.")
# -s /mnt/data3/yswang2024_data3/dataset/blended_downsample/5b08286b2775267d5b0634ba -m /mnt/data3/yswang2024_data3/experiment_output/pgsr_mini_5b08286b2775267d5b0634ba --data_device cpu --densify_abs_grad_threshold 0.0004