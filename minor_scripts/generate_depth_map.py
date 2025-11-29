import open3d as o3d
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import glob
import tqdm
from ysutils.util_mvsnet import read_pfm,load_cam

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

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--outdir', type=str, default=None)
    # parser.add_argument('--gtdir', type=str, default=None)
    # args = parser.parse_args()

    gtdir = '/mnt/data3/yswang2024_data3/dataset/blended_highres/scene'
    '/mnt/data3/yswang2024_data3/dataset/blended_highres/scene/5aa0f9d7a9efce63548c69a1/5aa0f9d7a9efce63548c69a1/5aa0f9d7a9efce63548c69a1/rendered_depth_maps/00000005.pfm'
    outdir = '/mnt/data3/yswang2024_data3/depth_sig_compare'

    scenelist = [
        '5bbb6eb2ea1cfa39f1af7e0c',
        '5bfc9d5aec61ca1dd69132a2',
        '5bfe5ae0fe0ea555e6a969ca',
        '5b08286b2775267d5b0634ba',
        '5aa515e613d42d091d29d300',
        '58eaf1513353456af3a1682a',
        '5b62647143840965efc0dbde',
        '5af02e904c8216544b4ab5a2',
    ]

    results_root = [
        '/mnt/data3/yswang2024_data3/gs2d_output',
        '/mnt/data3/yswang2024_data3/helixmini_output',
        '/mnt/data3/yswang2024_data3/pgsr_output',
        '/mnt/data3/yswang2024_data3/radegs_output',
    ]

    '/mnt/data3/yswang2024_data3/gs2d_output/gs2d_5af28cea59bc705737003253/gs2d_it30000/hypo_00000008.npy'
    '/mnt/data3/yswang2024_data3/helixmini_output/helixmini_5b69cc0cb44b61786eb959bf/helix_out/pgsr_it30000'
    '/mnt/data3/yswang2024_data3/pgsr_output/pgsr_5af28cea59bc705737003253/pgsr_it30000'
    '/mnt/data3/yswang2024_data3/radegs_output/radegs_5b6eff8b67b396324c5b2672/helix_out/pgsr_it30000'

    for scene in scenelist:
        os.makedirs(os.path.join(outdir, scene), exist_ok=True)
        print(scene)
        cams = glob.glob(os.path.join(gtdir, scene,scene,scene,'cams','*_cam.txt'))
        depths_gt = glob.glob(os.path.join(gtdir, scene,scene,scene,'rendered_depth_maps','*pfm'))
        hypos_gs2d = glob.glob(os.path.join('/mnt/data3/yswang2024_data3/gs2d_output',f'gs2d_{scene}', 'gs2d_it30000','hypo*'))
        hypos_helix = glob.glob(os.path.join('/mnt/data3/yswang2024_data3/helixmini_output', f'helixmini_{scene}', 'helix_out/pgsr_it30000','hypo*'))
        hypos_pgsr = glob.glob(os.path.join('/mnt/data3/yswang2024_data3/pgsr_output', f'pgsr_{scene}', 'pgsr_it30000','hypo*'))
        hypos_radegs = glob.glob(os.path.join('/mnt/data3/yswang2024_data3/radegs_output', f'radegs_{scene}', 'helix_out/pgsr_it30000','hypo*'))
        cams.sort()
        depths_gt.sort()
        hypos_gs2d.sort()
        hypos_helix.sort()
        hypos_pgsr.sort()
        hypos_radegs.sort()


        for i in tqdm.tqdm(range(len(depths_gt))):
            cam = load_cam(cams[i])[1]
            near = cam[3,0]
            far = cam[3,3]
            intrin = cam[:3,:3]
            hypo_gs2d = np.load(hypos_gs2d[i])
            hypo_pgsr = np.load(hypos_pgsr[i])
            hypo_radegs = np.load(hypos_radegs[i])
            hypo_helix = np.load(hypos_helix[i])
            depth_gt = read_pfm(depths_gt[i])[0]

            gt_h, gt_w = depth_gt.shape
            h,w,_ = hypo_helix.shape
            scale = gt_h / h
            intrin[:2,:] = intrin[:2,:] / scale
            depth_gt = cv2.resize(depth_gt,(w,h),interpolation=cv2.INTER_NEAREST)
            depth_gs2d = hypo2depth(hypo_gs2d,intrin).clip(near,far)
            depth_pgsr = hypo2depth(hypo_pgsr,intrin).clip(near,far)
            depth_radegs = hypo2depth(hypo_radegs,intrin).clip(near,far)
            depth_helix = hypo2depth(hypo_helix,intrin).clip(near,far)

            mask = depth_gt > 0.01
            err_gs2d = np.abs(depth_gs2d - depth_gt) * mask
            err_pgsr = np.abs(depth_pgsr - depth_gt) * mask
            err_helix = np.abs(depth_helix - depth_gt) * mask
            err_radegs = np.abs(depth_radegs - depth_gt) * mask

            err_near = 0
            err_far = (np.nanmean(err_helix) + np.nanmean(err_pgsr)) / 2 * 3
            err_gs2d = err_gs2d.clip(err_near,err_far)
            err_pgsr = err_pgsr.clip(err_near,err_far)
            err_helix = err_helix.clip(err_near,err_far)
            err_radegs = err_radegs.clip(err_near,err_far)

            depth_gt = np.asarray( (1 - (depth_gt - near)/(far-near)) * 255,dtype=np.uint8)
            depth_gs2d = np.asarray( (1 - (depth_gs2d - near)/(far-near)) * 255,dtype=np.uint8)
            depth_pgsr = np.asarray( (1 - (depth_pgsr - near)/(far-near)) * 255,dtype=np.uint8)
            depth_radegs = np.asarray( (1 - (depth_radegs - near)/(far-near)) * 255,dtype=np.uint8)
            depth_helix = np.asarray( (1 - (depth_helix - near)/(far-near)) * 255,dtype=np.uint8)
            color_depth_gt = cv2.applyColorMap(depth_gt,cv2.COLORMAP_JET)
            color_depth_gs2d = cv2.applyColorMap(depth_gs2d,cv2.COLORMAP_JET)
            color_depth_pgsr = cv2.applyColorMap(depth_pgsr,cv2.COLORMAP_JET)
            color_depth_radegs = cv2.applyColorMap(depth_radegs,cv2.COLORMAP_JET)
            color_depth_helix = cv2.applyColorMap(depth_helix,cv2.COLORMAP_JET)

            color_depth_gt = cv2.cvtColor(color_depth_gt,cv2.COLOR_BGR2RGB)
            color_depth_gs2d = cv2.cvtColor(color_depth_gs2d,cv2.COLOR_BGR2RGB)
            color_depth_pgsr = cv2.cvtColor(color_depth_pgsr,cv2.COLOR_BGR2RGB)
            color_depth_radegs = cv2.cvtColor(color_depth_radegs,cv2.COLOR_BGR2RGB)
            color_depth_helix = cv2.cvtColor(color_depth_helix,cv2.COLOR_BGR2RGB)


            err_gs2d = np.asarray( ((err_gs2d-err_near) / (err_far - err_near))* mask * 255,dtype=np.uint8)
            err_pgsr = np.asarray( ((err_pgsr-err_near) / (err_far - err_near))* mask * 255,dtype=np.uint8)
            err_helix = np.asarray( ((err_helix-err_near) / (err_far - err_near))* mask * 255,dtype=np.uint8)
            err_radegs = np.asarray( ((err_radegs-err_near) / (err_far - err_near))* mask * 255,dtype=np.uint8)
            color_err_gs2d = cv2.applyColorMap(err_gs2d,cv2.COLORMAP_JET)
            color_err_pgsr = cv2.applyColorMap(err_pgsr,cv2.COLORMAP_JET)
            color_err_helix = cv2.applyColorMap(err_helix,cv2.COLORMAP_JET)
            color_err_radegs = cv2.applyColorMap(err_radegs,cv2.COLORMAP_JET)

            # color_err_helix_trans = cv2.cvtColor(color_err_helix,cv2.COLOR_BGR2RGB)
            # plt.imshow(color_err_helix_trans)
            # plt.show()


            with open(os.path.join(outdir, scene,'range_rec.txt'),'a') as f:
                f.write(str(err_near))
                f.write(' ')
                f.write(str(err_far))
                f.write('\n')



            cv2.imwrite(os.path.join(outdir, scene,f'depth_gt_{i.__str__().zfill(4)}.png'),color_depth_gt)
            cv2.imwrite(os.path.join(outdir, scene,f'depth_pgsr_{i.__str__().zfill(4)}.png'),color_depth_pgsr)
            cv2.imwrite(os.path.join(outdir, scene,f'depth_radegs_{i.__str__().zfill(4)}.png'),color_depth_radegs)
            cv2.imwrite(os.path.join(outdir, scene,f'depth_helix_{i.__str__().zfill(4)}.png'),color_depth_helix)
            cv2.imwrite(os.path.join(outdir, scene,f'depth_gs2d_{i.__str__().zfill(4)}.png'),color_depth_gs2d)

            cv2.imwrite(os.path.join(outdir, scene,f'err_pgsr_{i.__str__().zfill(4)}.png'),color_err_pgsr)
            cv2.imwrite(os.path.join(outdir, scene,f'err_helix_{i.__str__().zfill(4)}.png'),color_err_helix)
            cv2.imwrite(os.path.join(outdir, scene,f'err_radegs_{i.__str__().zfill(4)}.png'),color_err_radegs)
            cv2.imwrite(os.path.join(outdir, scene,f'err_gs2d_{i.__str__().zfill(4)}.png'),color_err_gs2d)

