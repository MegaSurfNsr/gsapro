import argparse,sys,os
import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # Prevent processes being suspended
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from eval_utils import comput_one_scan_cuda, compute_scans, sava_result, limit_cpu_threads_used

# scans = [1,4,9,10,11,12,13,15,23,24,29,32,33,34,48,49,62,75,77,110,114,118]
parser = argparse.ArgumentParser()
parser.add_argument('--scans', type=str, default=None, help="scans to be evalutation")
parser.add_argument('--method', type=str, default='mvsnet', help="method name, such as mvsnet,casmvsnet")
parser.add_argument('--pred', type=str, default='./Predict/mvsnet', help="predict result ply")
parser.add_argument('--gt_dir', type=str, default='/mnt/data4/yswangdata4/dataset/DTU',help="groud truth ply file path")
parser.add_argument('--voxel_factor', type=float, default=1.28, help="voxel factor for alignment")
parser.add_argument('--down_dense', type=float, default=0.2, help="downsample density, Min dist between points when reducing")
parser.add_argument('--patch', type=float, default=60, help="patch size")
parser.add_argument('--max_dist', type=float, default=20, help="outlier thresshold of 20 mm")
parser.add_argument('--vis', action='store_true', help="visualization")
parser.add_argument('--vis_thresh', type=float, default=10, help="visualization distance threshold of 10mm")
parser.add_argument('--out_dir', type=str, default="./outputs", help="result save dir")
parser.add_argument('--save_file', type=str,required=True, help="save file path")
parser.add_argument('--num_workers', type=int, default=2, help="number of thread")
parser.add_argument('--device', type=int, default=0, help="cuda device id")
parser.add_argument('--model_name', type=str, default="", help="model name")
parser.add_argument('--cpu_percentage', type=float, default=0.1, help='percentage of cpu threads used')
parser.add_argument('--transform_2dgs', type=str, default=None, help="transform 2dgs results to dtu coord according to the cameras.npz file")
args = parser.parse_args()
# torch.cuda.set_device(args.device)



def eval_worker(args):
    scanid =args.scans_id
    pred_ply    = args.pred
    gt_ply      = os.path.join(args.gt_dir, f"Points/stl/stl{scanid:03}_total.ply")
    mask_file   = os.path.join(args.gt_dir, f'ObsMask/ObsMask{scanid}_10.mat')
    plane_file  = os.path.join(args.gt_dir, f'ObsMask/Plane{scanid}.mat')
    result = comput_one_scan_cuda(scanid, pred_ply, gt_ply, mask_file, plane_file,vis=args.vis,out_dir=args.out_dir, args=args)
    msg = "scan{}\t   acc = {:.4f}   comp = {:.4f}   overall = {:.4f}".format(scanid, result[0], result[1], result[2])
    print(msg)
    return result

def eval(testlist, args):
    gpu_used = min(args.num_workers, torch.cuda.device_count())
    if gpu_used > 1: args.num_workers = gpu_used
    print(f"Lets use {gpu_used} GPU with {args.num_workers} process for evaluation!")

    out = eval_worker(args)
    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, 'a') as f:
        f.write(args.method)
        f.write(' ')
        f.write(args.scans)
        f.write(' ')
        for rec in out:
            f.write(str(rec))
            f.write(' ')
        f.write('\n')



if __name__ == "__main__":
    # Limit the number of processes used in a server shared by multiple people to 
    # prevent excessive consumption of CPU resources. If the server is exclusively
    # owned by one person, this line can be commented out
    limit_cpu_threads_used(args.cpu_percentage)
    args.scans_id = int(args.scans[4:])
    print(args.scans_id)
    if args.num_workers >= 1:
        eval(args.scans_id, args)



