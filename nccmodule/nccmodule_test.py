import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from data_geoneus2acmh.module import *
from data_geoneus2acmh.module import read_dmb
import ncc

def read_img(root):
    imlist = []
    flist = glob.glob(os.path.join(root, 'images/*.jpg'))
    flist.sort()
    for f_im in flist:
        imlist.append(cv2.imread(f_im,cv2.IMREAD_GRAYSCALE))
    return np.ascontiguousarray(np.asarray(imlist))

def load_all_cams(root,changeTK=True):
    '''

    :param root:
    :param changeTK: change the idx of T and K. (From [n,2(T,K),4,4] to [n,2(K,T),4,4] )
    :return: np [n,2,4,4]
    '''
    cams = []
    flist = glob.glob(os.path.join(root, 'cams/*_cam.txt'))
    flist.sort()
    for f_cam in flist:
        temcam = np.zeros((2, 4, 4))
        cam = load_cam(f_cam)
        if changeTK:
            temcam[0] = cam[1]
            temcam[1] = cam[0]
        cams.append(temcam)
    return np.ascontiguousarray(np.asarray(cams,dtype=np.float32))

if __name__ == '__main__':
    root = "/home/yswang/Downloads/test/yswang_dtu24"
    ims = read_img(root)
    ims = np.asarray(ims/255.,dtype=np.float32) # 不做归一化可以大幅减少nan
    cams = load_all_cams(root)
    sfmlist = glob.glob(os.path.join(root,"sfm/*"))
    sfmlist.sort()

    print(cams.flags.c_contiguous)
    print(ims.flags.c_contiguous)

    n,h,w = ims.shape
    py_depths = np.zeros((n,h,w),dtype=np.float32)
    py_costs = np.ones((n,h,w),dtype=np.float32)
    py_planeHypos = np.zeros((n,h,w,4),dtype=np.float32)

    h_idx =np.arange(h,dtype=np.int32)
    w_idx =np.arange(w,dtype=np.int32)
    raw_grid = np.meshgrid(h_idx,w_idx,indexing="ij")
    coord_tem = np.ascontiguousarray(np.stack([raw_grid[0],raw_grid[1]],axis=-1).reshape(-1,2).copy()) #h,w

    print(coord_tem.dtype)
    coord_num_tem = coord_tem.shape[0]

    # usage
    # initialization
    ncc_module = ncc.NCCmodule()
    ncc_module.__version__()
    ncc_module.set_hw(h,w)
    # load the pair. info
    ncc_module.load_pair(root+"/pair.txt")

    # data preparation
    ncc_module.init(cams.data,ims.data,py_depths.data,py_costs.data,py_planeHypos.data)
    # Warning! If True here, the random init hypos would overwrite the existing hypos!
    ncc_module.set_init_flag(True)
    # If True, the return (checkerboard_hypos_cu) would convert to normal and depth, but you can still access the hypos from the hypos_mat_cu.
    ncc_module.set_normdepths_flag(True)
    # convert the numpy mat to instrinsic camera struct.
    ncc_module.genCamFromNp()
    # load data to cuda device.
    ncc_module.dataToCuda()

    # accessor for the data: ncc and plane hypos. The pointers do not change during the running.
    ncc_mat_cu = ncc_module.bind_ncc_cu()
    hypos_mat_cu = ncc_module.bind_hypos_cu()


    # new settings:
    ncc_module.set_back_propagation_flag(True)  # anywhere is ok
    ncc_module.ProcessCuSfm(sfmlist[0],0)  # sfmfile path, corresponding image idx
    ncc_module.set_init_flag(False)
    ncc_module.ProcessCuNcc(0, coord_tem, coord_num_tem)
    backgeo = ncc_module.bind_back_geo().cpu().reshape(1200,1600)
    backncc = ncc_module.bind_back_ncc().cpu().reshape(1200,1600)


    # ncc_module.ProcessCuSfm(sfmlist[1],15)  # sfmfile path, corresponding image idx
    # ncc_module.ProcessCuNcc(15, coord_tem, coord_num_tem)

    plt.imshow(ncc_mat_cu[0].cpu())
    plt.imshow(ncc_module.hypos_w2c(0).cpu()[:,:,-1].clip(400,1200))
    plt.imshow(backgeo.clip(0,10))
    plt.imshow(backncc)
    #

    ncc_module.ProcessCuSfm(sfmlist[1],15)  # sfmfile path, corresponding image idx
    ncc_module.set_init_flag(False)
    ncc_module.ProcessCuNcc(15, coord_tem, coord_num_tem)
    backgeo = ncc_module.bind_back_geo().cpu().reshape(1200,1600)
    backncc = ncc_module.bind_back_ncc().cpu().reshape(1200,1600)
    plt.imshow(backgeo.clip(0,10))
    plt.imshow(backncc)
    print("all done")




    #################
    #
    # coord_tem_rand = np.ascontiguousarray(coord_tem[np.random.permutation(coord_num_tem)])
    #
    #
    # if True:
    #     input_num=10000
    #     ncc_module.ProcessCuNcc(0, coord_tem_rand.data, input_num)
    #
    #     plt.figure()
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(hypos_mat_cu[0, :, :, :].cpu()[:, :, -1])
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(ncc_mat_cu[0].cpu())
    #     plt.show()
    #
    #     ncc_module.set_init_flag(False)
    #     from tqdm import tqdm
    #     bunch = int(np.floor(coord_num_tem / input_num))
    #
    #     displaycount = 1
    #     count = 0
    #     for i in tqdm(range(300000)):
    #         set = i % (bunch-1)
    #         if set == 0:
    #             coord_tem_rand = np.ascontiguousarray(coord_tem[np.random.permutation(coord_num_tem)])
    #             print("shuffle data")
    #         coord = coord_tem_rand[set*input_num:(set+1)*input_num]
    #         ncc_module.ProcessCuNcc(0, coord.data, input_num)
    #         if set == 0:
    #             count = count + 1
    #             if count == displaycount:
    #                 if count <=512:
    #                     displaycount = displaycount * 2
    #                 else:
    #                     displaycount = displaycount + 10
    #
    #                 plt.figure()
    #                 plt.subplot(1,2,1)
    #                 plt.imshow(hypos_mat_cu[0, :, :, :].cpu()[:,:,-1])
    #                 plt.subplot(1,2,2)
    #                 plt.imshow(ncc_mat_cu[0].cpu())
    #                 plt.show()
    #                 # checkerboard_ncc_cu = ncc_module.bind_checkerboard_ncc_cu()
    #                 # checkerboard_hypos_cu = ncc_module.bind_checkerboard_hypos_cu()
    #
    #
    # ### patchbased
    #
    #
    #
    #
    #
    #
    # # The pointers of checkerboard_ncc_cu & checkerboard_hypos_cu would change every iter! Please do not repeatedly access these pointers.
    #
    # ncc_module.ProcessCuNcc(0, coord_tem.data, coord_num_tem)
    # checkerboard_ncc_cu = ncc_module.bind_checkerboard_ncc_cu()
    # checkerboard_hypos_cu = ncc_module.bind_checkerboard_hypos_cu()
    #
    #
    # # c = ncc_mat_cu[0].cpu()
    # # plt.imshow(c)
    # c = checkerboard_ncc_cu.cpu()
    # plt.imshow(c.reshape(h,w,9)[:,:,8])
    # plt.show()
    # z = checkerboard_hypos_cu.cpu()
    # plt.imshow(z.reshape(h,w,9,4)[:,:,8,2])
    # plt.show()
    #
    #
    # print("all done")
    #
    # # check results
    # if False:
    #     import os.path
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import glob
    #     import cv2
    #     from data_geoneus2acmh.module import *
    #     results_root = "/home/yswang/Downloads/test"
    #     f1 = read_dmb(os.path.join(results_root,"temdepth_1.dmb")).reshape(1200,1600)
    #     f2 = read_dmb(os.path.join(results_root,"temcost_3.dmb")).reshape(1200,1600)
    #     plt.figure()
    #     plt.subplot(1,2,1)
    #     plt.imshow(f1)
    #     plt.subplot(1,2,2)
    #     plt.imshow(f2)
    #     img = cv2.imread("/home/yswang/Downloads/test/yswang_dtu24/images/00000015.jpg")
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     m = np.tile(np.expand_dims(f1<0.3,axis=-1),(1,1,3))
    #     plt.figure()
    #     plt.subplot(1,2,1)
    #     plt.imshow(img)
    #     plt.subplot(1,2,2)
    #     plt.imshow(img * m)
    #
    #
    #     f3 = read_dmb(os.path.join(results_root,"temcost_1.dmb")).reshape(1200,1600)
    #     f4 = read_dmb(os.path.join(results_root,"temdepth_1_saver.dmb")).reshape(1200,1600)
    #     plt.figure()
    #     plt.subplot(2,2,1)
    #     plt.imshow(f1)
    #     plt.subplot(2,2,2)
    #     plt.imshow(f2)
    #     plt.subplot(2,2,3)
    #     plt.imshow(f3)
    #     plt.subplot(2,2,4)
    #     plt.imshow(f4)
    #
    #
    #     results_root = "/home/yswang/Downloads/test"
    #     f1 = read_dmb(os.path.join(results_root,"backtest1/nobk","temdepth_2.dmb")).reshape(1200,1600)
    #     f2 = read_dmb(os.path.join(results_root,"backtest1/bk","temdepth_2.dmb")).reshape(1200,1600)
    #     f3 = read_dmb(os.path.join(results_root,"backtest1/nobk","temcost_1.dmb")).reshape(1200,1600)
    #     f4 = read_dmb(os.path.join(results_root,"backtest1/bk","temcost_1.dmb")).reshape(1200,1600)
    #     plt.figure()
    #     plt.subplot(2,2,1)
    #     plt.imshow(f1)
    #     plt.subplot(2,2,2)
    #     plt.imshow(f2)
    #     plt.subplot(2,2,3)
    #     plt.imshow(f3)
    #     plt.subplot(2,2,4)
    #     plt.imshow(f4)
    #     gt = cv2.imread("/home/yswang/Downloads/test/yswang_dtu24/depth_map_0015.pfm", -1)
    #
    #
    #     results_root = "/home/yswang/Downloads/test"
    #     d1 = read_dmb(os.path.join(results_root,"backtest1/nobk","temdepth_2.dmb")).reshape(1200,1600)
    #     d2 = read_dmb(os.path.join(results_root,"backtest1/bk","temdepth_2.dmb")).reshape(1200,1600)
    #     gt = cv2.imread("/home/yswang/Downloads/test/yswang_dtu24/depth_map_0015.pfm",-1)
    #     mask=cv2.imread("/home/yswang/Downloads/MVSDataset/neus_data/data_DTU/dtu_scan24/mask/015.png",-1)[:,:,0] > 0
    #     mask = np.logical_and(mask,gt>0)
    #     print((np.abs(d1-gt) * mask).sum()/mask.sum()) #squre
    #     print((np.abs(d2-gt) * mask).sum()/mask.sum())
    #
    #
    #     plt.figure()
    #     plt.subplot(1,2,1)
    #     e1 = np.asarray(np.abs(d1 - gt).clip(0, 10) / 10 * mask * 255, dtype=np.uint8)
    #     plt.imshow(cv2.cvtColor(cv2.applyColorMap(e1,cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB))
    #     plt.subplot(1,2,2)
    #     e2 = np.asarray(np.abs(d2 - gt).clip(0, 10) / 10 * mask * 255, dtype=np.uint8)
    #     plt.imshow(cv2.cvtColor(cv2.applyColorMap(e2,cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB))
    #
    #
    #
    #     gt = cv2.imread("/home/yswang/Downloads/test/yswang_dtu24/depth_map_0015.pfm",-1)
    #     results_root = "/home/yswang/Downloads/test"
    #     f1 = results_root + "/backgeo_900_saver.dmb"
    #     f2 = results_root + "/backncc_900_saver.dmb"
    #     d1 = results_root + "/temcost_1.dmb"
    #     d2 = results_root + "/temdepth_2.dmb"
    #     plt.figure()
    #     plt.subplot(2,2,1)
    #     plt.imshow(read_dmb(f1).reshape(1200,1600))
    #     plt.subplot(2,2,2)
    #     plt.imshow(read_dmb(f2).reshape(1200,1600))
    #     plt.subplot(2,2,3)
    #     plt.imshow(read_dmb(d1).reshape(1200,1600))
    #     plt.subplot(2,2,4)
    #     plt.imshow(read_dmb(d2).clip(0,50).reshape(1200,1600))
    #
    #     c = read_dmb(d1).reshape(1200,1600)
    #     d = read_dmb(d2).reshape(1200,1600)
    #     plt.imshow(np.abs(gt - d).clip(0,10))
    #     mask=cv2.imread("/home/yswang/Downloads/MVSDataset/neus_data/data_DTU/dtu_scan24/mask/015.png",-1)[:,:,0] > 0
    #     mask = np.logical_and(mask,gt>0)
    #     print((np.abs(d-gt) * mask).sum()/mask.sum()) #squre
    #     # print((np.abs(d2-gt) * mask).sum()/mask.sum())
    #
    #
    #
    #
    # depth = read_dmb(d2)
    # depth = depth.reshape(h,w)
    # mask = None
    # import open3d as o3d
    #
    # def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    #     i , j = torch.meshgrid(
    #     torch.linspace(0 , W-1, W, device=c2w.device),
    #     torch.linspace(0 , H-1, H, device=c2w.device))
    #     # pytorch's meshgrid has indexing='ij'
    #     i = i.t().float()
    #     j = j.t().float()
    #     if mode == 'lefttop':
    #         pass
    #     elif mode == 'center':
    #         i, j = i+0.5, j+0.5
    #     elif mode == 'random':
    #         i = i+torch.rand_like(i)
    #         j = j+torch.rand_like(j)
    #     else:
    #         raise NotImplementedError
    #     if flip_x:
    #         i = i.flip((1,))
    #     if flip_y:
    #         j = j.flip((0,))
    #     if inverse_y:
    #         dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    #     else:
    #         dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    #     # Rotate ray directions from camera frame to the world frame
    #     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    #     # Translate camera frame's origin to the world frame. It is the origin of all rays.
    #     rays_o = c2w[:3,3].expand(rays_d.shape)
    #     return rays_o, rays_d
    #
    #
    # def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    #
    #     rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y,
    #                               mode=mode)
    #     viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    #     return rays_o, rays_d, viewdirs
    #
    #
    # def depth_to_points(depth, mask, c2w, K):
    #     H, W = depth.shape
    #     rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K, c2w, False, True, False, False)
    #     points = rays_o[mask].cpu() + viewdirs[mask].cpu() * depth[..., None][mask] * torch.norm(rays_d, dim=2)[...,None][mask]
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points.reshape(-1,3).cpu().numpy())
    #     o3d.io.write_point_cloud('depth_pts.ply', pcd)
    #
    #
    # depth_to_points(depth, mask,
    #                torch.tensor(
    #     np.linalg.inv(cams[0, 1, ...]))[:3, :4], torch.tensor(cams[0, 0, :3, :3]))
    #
    #
    #

