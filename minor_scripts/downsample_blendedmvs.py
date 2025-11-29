'/mnt/data3/yswang2024_data3/dataset/blended_highres/scene'
'/mnt/data3/yswang2024_data3/dataset/blended_mesh'
import open3d as o3d
import os
import glob
import shutil
from ysutils.util_mvsnet import load_cam, write_cam
import tqdm
import cv2
import matplotlib.pyplot as plt

inroot = '/mnt/data3/yswang2024_data3/dataset/blended_highres/scene'
outroot = '/mnt/data3/yswang2024_data3/dataset/blended_downsample'

with open('/mnt/data3/yswang2024_data3/dataset/blended_highres/process_scene2','r') as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines]

for scene in lines:
    indir = os.path.join(inroot, scene,scene,scene)
    outdir = os.path.join(outroot, scene)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir,'cams'), exist_ok=True)
    os.makedirs(os.path.join(outdir,'images'), exist_ok=True)
    shutil.copyfile(os.path.join(indir,'cams','pair.txt'),os.path.join(outdir,'pair.txt'))
    shutil.copyfile(os.path.join(indir,'points.ply'),os.path.join(outdir,'points.ply'))
    camslist = glob.glob(os.path.join(indir,'cams','*_cam.txt'))
    imglist = glob.glob(os.path.join(indir,'blended_images','*.jpg'))
    camslist.sort()
    imglist.sort()
    print(f'{scene}')
    for camfile in camslist:
        cam = load_cam(camfile)
        extrinsic = cam[0]
        intrinsic = cam[1][:3,:3]
        intrinsic[:2,:] = intrinsic[:2,:]/2
        depth_stat = cam[1][3,:]
        outcamfile = os.path.join(outdir,'cams',os.path.basename(camfile))
        write_cam(outcamfile, intrinsic, extrinsic, depth_stat)

    for imgfile in tqdm.tqdm(imglist):
        img = cv2.imread(imgfile)
        h,w,_ = img.shape
        img = cv2.resize(img,(w//2,h//2))
        outimgfile = os.path.join(outdir,'images',os.path.basename(imgfile))
        # imgout = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(outimgfile,img)



