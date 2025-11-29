import torch
import numpy as np
import sys
import argparse
import errno, os
import glob
import os.path as osp
import re
# import matplotlib.pyplot as plt
import cv2
from PIL import Image
import gc
import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
from warp_func import *

# /mnt/data3/yswang2024_data3/experiment_output/helixgau_v1
# /mnt/data2/yswang2024/dataset/scan24_2dgs/cams
# /mnt/data2/yswang2024/dataset/scan24_2dgs/images_acmh
# /mnt/data3/yswang2024_data3/experiment_output
parser = argparse.ArgumentParser(description='Depth fusion with consistency check.')
parser.add_argument('--cam_path', type=str,
                    default=None)
parser.add_argument('--image_path', type=str,
                    default=None)
parser.add_argument('--helixout_path', type=str,
                    default=None)
parser.add_argument('--save_path', type=str,
                    default=None)
parser.add_argument('--dist_thresh', type=float, default=0.05)  # 0.1 blended  # ncc 0.05
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--dist_thresh_downsample', type=float, default=0.05)
parser.add_argument('--extract_color', action='store_true')
parser.add_argument('--prob_thresh', type=float, default=0.35)
parser.add_argument('--num_consist', type=int, default=4)  # 5 blended # ncc 10
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--rand_downsample', action='store_true')
parser.add_argument('--onlyhypo', action='store_true')

args = parser.parse_args()

MAX_POINTS = 10000000


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()


def write_ply(file, points, rgb_flag=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if rgb_flag:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)
    o3d.io.write_point_cloud(file, pcd, write_ascii=False)


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''

    :param ref_depth: (1, 1, H, W)
    :param src_depths: (B, 1, H, W)
    :param ref_proj: (1, 4, 4)
    :param src_proj: (B, 4, 4)
    :return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
    '''

    ref_pc = generate_points_from_depth(ref_depth, ref_proj)
    src_pcs = generate_points_from_depth(src_depths, src_projs)

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

    # _, axs = plt.subplots(3, 4)
    # for i in range(3):
    # 	axs[i, 0].imshow(src_pcs[0, i], vmin=0, vmax=1)
    # 	axs[i, 1].imshow(aligned_pcs[0, i],  vmin=0, vmax=1)
    # 	axs[i, 2].imshow(ref_pc[0, i],  vmin=0, vmax=1)
    # 	axs[i, 3].imshow(ref_pc[0, i] - aligned_pcs[0, i], vmin=-0.5, vmax=0.5, cmap='coolwarm')
    # plt.show()

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist


def parse_cameras(path):
    cam_txt = open(path).readlines()
    f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5])
    intr_mat = f(cam_txt[7:10])

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat


def plane_to_depth(hypo, K):
    pass


def load_data(cam_path, helixout_path, image_path, rgb_flag=False):
    '''

    :param root_path:
    :param scene_name:
    :param thresh:
    :return: depth
    '''
    rgb_paths = sorted(glob.glob(os.path.join(image_path, '*')))
    cam_paths = sorted(glob.glob(os.path.join(cam_path, '*')))
    helixout_paths_raw = sorted(glob.glob(os.path.join(helixout_path, 'hypo*')))
    helixout_costs_raw = sorted(glob.glob(os.path.join(helixout_path, 'cost*')))
    helixout_paths = []
    helixcost_paths = []

    for h in helixout_paths_raw:
        if len(h.split('/')[-1])==17:
            helixout_paths.append(h)
    for h in helixout_costs_raw:
        if len(h.split('/')[-1])==17:
            helixcost_paths.append(h)
    helixout_paths.sort()
    helixcost_paths.sort()
    depths = []
    projs = []
    rgbs = []
    confs = []
    norms = []
    if len(rgb_paths) == len(helixout_paths):
        args.prefix = 'nocost_'
    for i in tqdm.tqdm(range(len(rgb_paths))):
        extr_mat, intr_mat = parse_cameras(cam_paths[i])
        if args.scale != 1:
            intr_mat[:2, :] = intr_mat[:2, :] / args.scale

        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))

        conf_map = np.load(helixcost_paths[i])
        helixout = np.load(helixout_paths[i])

        h, w, _ = conf_map.shape
        confs.append(torch.from_numpy(conf_map))

        h_idx = np.arange(h, dtype=np.int32)
        w_idx = np.arange(w, dtype=np.int32)
        raw_grid = list(np.meshgrid(h_idx, w_idx, indexing="ij"))
        raw_grid.append(np.ones([h, w]))
        dep_map = -1 * helixout[:, :, -1] * intr_mat[0, 0] / (
                    (raw_grid[1] - intr_mat[0, 2]) * helixout[:, :, 0] + (intr_mat[0, 0] / intr_mat[1, 1]) * (
                        raw_grid[0] - intr_mat[1, 2]) * helixout[:, :, 1] + intr_mat[0, 0] * helixout[:, :, 2])
        depths.append(torch.from_numpy(dep_map).unsqueeze(0))
        norm = helixout[:, :, :3]
        norm = norm / np.linalg.norm(norm, axis=-1, keepdims=True)
        norms.append(
            torch.from_numpy(np.linalg.pinv(extr_mat[:3, :3])[None, None, :, :] @ norm[:, :, :, None]).squeeze(-1))
        if rgb_flag:
            rgb = np.array(Image.open(rgb_paths[i])) / 256.
            h_rgb, w_rgb, _ = rgb.shape
            if h_rgb != h:
                rgb = cv2.resize(rgb, (w, h))
            rgbs.append(rgb)

    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()
    confs = torch.stack(confs).float()
    if args.device == 'cuda' and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()
        confs = confs.cuda()

    return depths, projs, rgbs, confs, norms

def load_data_dtu(cam_path, helixout_path, image_path, rgb_flag=False,i3flag=False):
    '''

    :param root_path:
    :param scene_name:
    :param thresh:
    :return: depth
    '''
    rgb_paths = sorted(glob.glob(os.path.join(image_path, '*_3_r5000.png')))
    cam_paths = sorted(glob.glob(os.path.join(cam_path, '*_cam.txt')))
    helixout_paths_raw = sorted(glob.glob(os.path.join(helixout_path, 'depth*')))
    helixout_costs_raw = sorted(glob.glob(os.path.join(helixout_path, 'cost*')))
    helixout_paths = []
    helixcost_paths = []

    if i3flag:
        for h in helixout_paths_raw:
            if len(h.split('/')[-1]) > 18:
                helixout_paths.append(h)
        for h in helixout_costs_raw:
            if len(h.split('/')[-1]) > 17:
                helixcost_paths.append(h)
    else:
        for h in helixout_paths_raw:
            if len(h.split('/')[-1])==18:
                helixout_paths.append(h)
        for h in helixout_costs_raw:
            if len(h.split('/')[-1])==17:
                helixcost_paths.append(h)

    helixout_paths.sort()
    helixcost_paths.sort()
    depths = []
    projs = []
    rgbs = []
    confs = []
    norms = []

    for i in tqdm.tqdm(range(len(rgb_paths))):
        extr_mat, intr_mat = parse_cameras(cam_paths[i])
        if args.scale != 1:
            intr_mat[:2, :] = intr_mat[:2, :] / args.scale

        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4])
        projs.append(torch.from_numpy(proj_mat))

        helixout = np.load(helixout_paths[i])
        conf_map = np.load(helixcost_paths[i])
        h, w, _ = conf_map.shape
        confs.append(torch.from_numpy(conf_map))


        dep_map = helixout
        depths.append(torch.from_numpy(dep_map).unsqueeze(0))

    depths = torch.stack(depths).float()
    projs = torch.stack(projs).float()
    confs = torch.stack(confs).float()
    if args.device == 'cuda' and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()
        confs = confs.cuda()

    return depths, projs, rgbs, confs, norms
# depths torch.Size([49, 1, 600, 800])



def extract_points(pc, mask, norm=None, rgb=None):
    pc = pc.cpu().numpy()
    mask = mask.cpu().numpy()
    norm = norm.cpu().numpy()

    mask = np.reshape(mask, (-1,))
    pc = np.reshape(pc, (-1, 3))
    norm = np.reshape(norm, (-1, 3))

    tem_mask = np.logical_not(np.isnan(pc.max(-1)))
    mask = np.logical_and(mask, tem_mask)
    points = pc[np.where(mask)]
    norms = norm[np.where(mask)]

    if rgb is not None:
        rgb = np.reshape(rgb, (-1, 3))
        colors = rgb[np.where(mask)]
        points_with_color = np.concatenate([points, colors], axis=1)
        return points_with_color, norms
    else:
        return points, norms


def main():
    print(args.helixout_path)
    acc_pthred = 0.005

    dtupm_dir = '/mnt/data4/yswangdata4/dataset/dtu_patchmatch'
    dtugt_dir = '/mnt/data4/yswangdata4/dataset/DTU/datasetPM'

    dtu_scenelist = glob.glob('/mnt/data4/yswangdata4/dataset/dtu_patchmatch/s*')
    dtu_scenelist.sort()
    dtu_scenelist = [os.path.basename(s) for s in dtu_scenelist]

    # args.dist_thresh = 0.05
    # args.dist_thresh_downsample = 0.05


    # for dtu
    for scene in dtu_scenelist:
        args.dist_thresh = 0.2
        args.dist_thresh_downsample = 0.2
        dtu_save_dir = os.path.join(dtupm_dir, scene,'patchmatch')
        print(scene)
        cams_path = os.path.join(dtugt_dir,'Cameras')
        image_path = os.path.join(dtugt_dir,'Rectified',scene)
        hypos_path = os.path.join(dtupm_dir,scene, 'patchmatch')
        cmd = [cams_path,image_path,hypos_path,dtu_save_dir]

        args.cam_path = cmd[0]
        args.image_path = cmd[1]
        args.helixout_path = cmd[2]
        args.save_path = cmd[3]
        args.extract_color = False


        depths, projs, rgbs, confs, norms = load_data_dtu(args.cam_path, args.helixout_path, args.image_path,
                                                      args.extract_color)


        tot_frame = depths.shape[0]
        height, width = depths.shape[2], depths.shape[3]

        for i in tqdm.tqdm(range(tot_frame)):
            pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype)
            val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype)
            j = 0
            batch_size = 20

            while True:
                ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i + 1],
                                                 src_depths=depths[j:min(j + batch_size, tot_frame)],
                                                 ref_proj=projs[i:i + 1], src_projs=projs[j:min(j + batch_size, tot_frame)])
                masks = (dist < args.dist_thresh).float()
                masked_pc = pcs * masks
                pc_buff += masked_pc.sum(dim=0, keepdim=False)
                val_cnt += masks.sum(dim=0, keepdim=False)

                j += batch_size
                if j >= tot_frame:
                    break

            final_mask = (val_cnt >= args.num_consist).squeeze(0) * (confs[i].squeeze(-1) < args.prob_thresh)
            cv2.imwrite(os.path.join(dtu_save_dir, f'mask_{i.__str__().zfill(8)}.png'), (final_mask*255).cpu().numpy().astype(np.uint8))
        # write_ply('{}/{}.ply'.format(args.save_path, 'fusion'), np.concatenate(points, axis=0),args.extract_color)
        del depths, rgbs, projs, confs, norms
        gc.collect()


        depths, projs, rgbs, confs, norms = load_data_dtu(args.cam_path, args.helixout_path, args.image_path,
                                                      args.extract_color,i3flag=True)


        tot_frame = depths.shape[0]
        height, width = depths.shape[2], depths.shape[3]

        for i in tqdm.tqdm(range(tot_frame)):
            pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype)
            val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype)
            j = 0
            batch_size = 20

            while True:
                ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i + 1],
                                                 src_depths=depths[j:min(j + batch_size, tot_frame)],
                                                 ref_proj=projs[i:i + 1], src_projs=projs[j:min(j + batch_size, tot_frame)])
                masks = (dist < args.dist_thresh).float()
                masked_pc = pcs * masks
                pc_buff += masked_pc.sum(dim=0, keepdim=False)
                val_cnt += masks.sum(dim=0, keepdim=False)

                j += batch_size
                if j >= tot_frame:
                    break

            final_mask = (val_cnt >= args.num_consist).squeeze(0) * (confs[i].squeeze(-1) < args.prob_thresh)
            cv2.imwrite(os.path.join(dtu_save_dir, f'mask_{i.__str__().zfill(8)}_i3.png'), (final_mask*255).cpu().numpy().astype(np.uint8))
        # write_ply('{}/{}.ply'.format(args.save_path, 'fusion'), np.concatenate(points, axis=0),args.extract_color)
        del depths, rgbs, projs, confs, norms
        gc.collect()

def merge(root_path, ):
    all_scenes = open(args.data_list, 'r').readlines()
    all_scenes = list(map(str.strip, all_scenes))
    for scene in all_scenes:
        mkdir_p('{}/{}'.format(args.save_path, scene))
        points = []
        paths = sorted(glob.glob('{}/{}/*.npy'.format(root_path, scene, )))
        for p in paths:
            points.append(np.load(p))
        points = np.concatenate(points, axis=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.)

        o3d.io.write_point_cloud("{}/{}.ply".format(args.save_path, scene), pcd, write_ascii=False)
        print('Save {}/{}.ply successful.'.format(args.save_path, scene))


if __name__ == '__main__':
    with torch.no_grad():
        main()
