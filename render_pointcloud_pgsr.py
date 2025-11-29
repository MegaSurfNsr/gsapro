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

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque
import torch.nn as nn
import matplotlib.pyplot as plt
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
class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = list(np.meshgrid(range(self.width), range(self.height), indexing='xy'))
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background,
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False, voxel_size=0.05):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    max_depth = scene.cameras_extent * 2

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(os.path.join(model_path, f'pgsr_it{iteration}'),exist_ok=True)
    makedirs(os.path.join(model_path, f'pgsr_it{iteration}_depth'),exist_ok=True)

    depths_tsdf_fusion = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()

        np.save(os.path.join(model_path, f'pgsr_it{iteration}_depth', f'depth_{view.image_name}'), depth)

        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = out["rendered_normal"].permute(1, 2, 0)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)
        normal = normal.detach().cpu().numpy()
        K = view.get_k().detach().cpu().numpy()

        hypo = generate_plane_hypo(depth, normal, K)
        np.save(os.path.join(model_path, f'pgsr_it{iteration}', f'hypo_{view.image_name}'), hypo)


        normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)


        if name == 'test':
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (
                        rendering.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
                np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)
    #
    #     if use_depth_filter:
    #         view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
    #         depth_normal = out["depth_normal"].permute(1, 2, 0)
    #         depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
    #         dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
    #         angle = torch.acos(dot)
    #         mask = angle > (80.0 / 180 * 3.14159)
    #         depth_tsdf[mask] = 0
    #     depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())
    #
    # points_w = []
    # h,w = depths_tsdf_fusion[0].shape
    # bproj = BackprojectDepth(1,h,w)
    # pcd = o3d.geometry.PointCloud()
    #
    # for idx, view in enumerate(tqdm(views, desc="Fusion progress")):
    #     ref_depth = depths_tsdf_fusion[idx]
    #     K= np.eye(3)
    #     K[0,0],K[1,1],K[0,2],K[1,2] = view.Fx, view.Fy, view.Cx, view.Cy
    #
    #     if view.mask is not None:
    #         ref_depth[view.mask.squeeze() < 0.5] = 0
    #     ref_depth[ref_depth > max_depth] = 0
    #     ref_depth = ref_depth.detach().cpu()
    #
    #     extrin = np.identity(4)
    #     extrin[:3, :3] = view.R.transpose(-1, -2)
    #     extrin[:3, 3] = view.T
    #     # pose = np.linalg.inv(extrin)
    #
    #     points_c = bproj(ref_depth.unsqueeze(dim=0).unsqueeze(dim=0).float(),
    #                      torch.from_numpy(np.linalg.pinv(K)).unsqueeze(dim=0).float())
    #     points_w.append(np.asarray(np.matmul(np.linalg.pinv(extrin),points_c[0])[:3, :]))
    #
    #     if idx % 20 == 0 or idx == len(views) - 1:
    #         print('down sampling point cloud')
    #         points_wc = np.concatenate(points_w, axis=1)
    #         points = points_wc[:, np.logical_not(np.isnan(points_wc.sum(axis=0)))]
    #         pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd.points),points.transpose()],axis=0))
    #         pcd = pcd.voxel_down_sample(voxel_size)
    return 0


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                max_depth: float, voxel_size: float, num_cluster: int, use_depth_filter: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        if not skip_train:
            pcd = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians,
                       pipeline, background,
                       max_depth=max_depth, voxel_size=voxel_size,use_depth_filter=use_depth_filter)

            # path = os.path.join(dataset.model_path, "mesh")
            # os.makedirs(path, exist_ok=True)
            # o3d.io.write_point_cloud(os.path.join(path, "tsdf_fusion.ply"), pcd)




if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.05, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)