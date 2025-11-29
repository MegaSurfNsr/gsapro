import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino.layers.attention import get_attention_type, MemEffAttention, FlashAttention2, CrossLinearAttention
from models.dino.layers.block import CrossBlock
from models.dino.layers.mlp import Mlp
from models.dino.layers.swiglu_ffn import SwiGLU

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='IN', **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            pad: int = 1,
            dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, in_channels=3, norm_type='BN'):
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(in_channels, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type)

        self.downsample1 = Conv2d(feat_chs[0], feat_chs[1], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv10 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)
        self.conv11 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)

        self.downsample2 = Conv2d(feat_chs[1], feat_chs[2], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv20 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)
        self.conv21 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)

        self.downsample3 = Conv2d(feat_chs[2], feat_chs[3], 3, stride=2, padding=1, norm_type=norm_type)
        self.conv30 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)
        self.conv31 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)

    def forward(self, x):
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        down_conv0 = self.downsample1(conv01)
        conv10 = self.conv10(down_conv0)
        conv11 = self.conv11(conv10)
        down_conv1 = self.downsample2(conv11)
        conv20 = self.conv20(down_conv1)
        conv21 = self.conv21(conv20)
        down_conv2 = self.downsample3(conv21)
        conv30 = self.conv30(down_conv2)
        conv31 = self.conv31(conv30)

        return [conv01, conv11, conv21, conv31]


class FPNDecoder(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder, self).__init__()
        final_ch = feat_chs[-1]
        self.out0 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[3], kernel_size=1), nn.BatchNorm2d(feat_chs[3]), Swish())

        self.inner1 = nn.Conv2d(feat_chs[2], final_ch, 1)
        self.out1 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[2], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[2]), Swish())

        self.inner2 = nn.Conv2d(feat_chs[1], final_ch, 1)
        self.out2 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[1], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[1]), Swish())

        self.inner3 = nn.Conv2d(feat_chs[0], final_ch, 1)
        self.out3 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[0], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[0]), Swish())

    def forward(self, conv01, conv11, conv21, conv31):
        intra_feat = conv31
        out0 = self.out0(intra_feat)

        intra_feat = F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv21)
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv11)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(conv01)
        out3 = self.out3(intra_feat)

        return [out0, out1, out2, out3]


class CrossVITDecoder(nn.Module):
    def __init__(self, args):
        super(CrossVITDecoder, self).__init__()
        self.dino_cfg = args['dino_cfg']
        self.decoder_cfg = args['dino_cfg']['decoder_cfg']
        attention_class = get_attention_type(self.decoder_cfg['attention_type'])

        ffn_type = self.decoder_cfg.get("ffn_type", "ffn")
        if ffn_type == "ffn":
            ffn_class = Mlp
        elif ffn_type == "glu":
            ffn_class = SwiGLU
        else:
            raise NotImplementedError(f"Unknown FFN...{ffn_type}")

        self.self_cross_types = self.decoder_cfg.get("self_cross_types", None)

        if self.self_cross_types is not None:
            self_attn_class = get_attention_type(self.self_cross_types[0])
            cross_attn_class = get_attention_type(self.self_cross_types[1])
        else:
            self_attn_class = attention_class
            cross_attn_class = attention_class

        self.self_attn_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()

        self.no_combine_norm = self.decoder_cfg.get("no_combine_norm", False)
        if not self.no_combine_norm:
            self.norm_layers = nn.ModuleList()

        self.prev_values = nn.ParameterList()
        for _ in range(self.dino_cfg['cross_interval_layers'] - 1):
            self.self_attn_blocks.append(CrossBlock(dim=self.decoder_cfg['d_model'], num_heads=self.decoder_cfg['nhead'],
                                                    attn_class=self_attn_class, ffn_layer=ffn_class, **self.decoder_cfg))
            if not self.no_combine_norm:
                self.norm_layers.append(nn.LayerNorm(self.decoder_cfg['d_model'], eps=1e-6))
            self.prev_values.append(nn.Parameter(torch.tensor(self.decoder_cfg['prev_values']), requires_grad=True))
        for _ in range(self.dino_cfg['cross_interval_layers']):
            self.cross_attn_blocks.append(CrossBlock(dim=self.decoder_cfg['d_model'], num_heads=self.decoder_cfg['nhead'],
                                                     attn_class=cross_attn_class, ffn_layer=ffn_class, **self.decoder_cfg))

        ch, vit_ch = args['out_ch'], args['vit_ch']
        self.proj = nn.Sequential(nn.Conv2d(vit_ch, ch * 4, 3, stride=1, padding=1),
                                  nn.BatchNorm2d(ch * 4), nn.SiLU())
        self.upsampler0 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(ch * 2), nn.SiLU())
        self.upsampler1 = nn.Sequential(nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x, Fmats=None, vit_shape=None):  # [B,V,HW,C]*N, Fmats:[B,V-1,HW,HW]
        B, V, H, W, C = vit_shape
        # x:[B,V,HW,C]*N

        src_feat = None
        ref_feat_list, src_feat_list = [], []

        # [B,V,HW,C]*N
        for v in range(V):
            if v == 0:  # self-attention (ref)
                for i in range(len(self.self_attn_blocks) + 1):
                    if i == 0:
                        ref_feat_list.append(x[i][:, v])
                    else:
                        attn_inputs = {'x': ref_feat_list[-1]}
                        pre_ref_feat = self.self_attn_blocks[i - 1](**attn_inputs)
                        new_ref_feat = self.prev_values[i - 1] * pre_ref_feat + x[i][:, v]  # AAS
                        if not self.no_combine_norm:
                            new_ref_feat = self.norm_layers[i - 1](new_ref_feat)
                        ref_feat_list.append(new_ref_feat)
            else:  # cross-attention (src)
                for i in range(len(self.cross_attn_blocks)):
                    if i == 0:
                        attn_inputs = {'x': x[i][:, v], 'key': ref_feat_list[i], 'value': ref_feat_list[i]}
                    else:
                        query = self.prev_values[i - 1] * src_feat + x[i][:, v]
                        if not self.no_combine_norm:
                            query = self.norm_layers[i - 1](query)
                        attn_inputs = {'x': query, 'key': ref_feat_list[i], 'value': ref_feat_list[i]}

                    src_feat = self.cross_attn_blocks[i](**attn_inputs)
                src_feat_list.append(src_feat.unsqueeze(1))

        src_feat = torch.cat(src_feat_list, dim=1)  # [B,V-1,HW,C]
        x = torch.cat([ref_feat_list[-1].unsqueeze(1), src_feat.reshape(B, V - 1, H * W, C)], dim=1)  # [B,V,HW,C]
        x = x.reshape(B * V, H, W, C).permute(0, 3, 1, 2)  # [BV,C,H,W]

        x = self.proj(x)
        x = self.upsampler0(x)
        x = self.upsampler1(x)

        return x

def warp_feat(ref_fea, src_proj_new, ref_proj_new,src_depmap, feat_add=None, return_coord=False,return_warp_depth=False):
    # warp src to ref according depth map
    # src_fea: [B, C, H, W]
    # src_K: [B, 3, 3]
    # src_T: [B, 3, 4]
    # depth: b ,c, h , w
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    if len(src_depmap.shape) == 3:
        src_depmap = src_depmap[:,:,:,None]
    if len(src_depmap.shape) == 4:
        src_depmap = src_depmap.permute(0, 2, 3, 1)

    batch, channels = ref_fea.shape[0], ref_fea.shape[1]
    height, width = ref_fea.shape[2], ref_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(ref_proj_new, torch.inverse(src_proj_new))
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=ref_fea.device)])

        y, x = y.repeat(batch,1,1).contiguous(), x.repeat(batch,1,1).contiguous()
        xyz = torch.stack((x,y, torch.ones_like(x)), dim=-1) * src_depmap

        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        rot_xyz = torch.matmul(rot[:,None,None,:,:], xyz[:,:,:,:,None])
        proj_xyz = rot_xyz + trans[:,None,None,:,:]

        if return_warp_depth:
            warp_depth = proj_xyz[:, :, :, 2,0].detach().clone()
        else:
            warp_depth = None
        proj_xy = proj_xyz[:, :, :, :2,0] / proj_xyz[:, :, :, 2,0,None]  # [b,h,w,2(x,y)]

        if return_coord:
            coord = proj_xy.detach().clone()
        else:
            coord = None

        proj_x_normalized = proj_xy[:, :, :, 0] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, :, :, 1] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)

    warped_src_fea = F.grid_sample(ref_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    if feat_add is not None:
        warped_src_fea_add = F.grid_sample(feat_add, grid.view(batch, height, width, 2), mode='bilinear',
                                       padding_mode='zeros')
    else:
        warped_src_fea_add = None

    return {
        'warped_src_fea': warped_src_fea,
        'warped_src_fea_add': warped_src_fea_add,
        'coord': coord,
        'warp_depth': warp_depth
    }



    # plt.imshow(warped_src_fea[0].cpu().permute(1,2,0))
    # plt.show()


def warp_coord(src_proj_new, ref_proj_new, src_depmap):
    # warp src to ref according depth map
    # src_fea: [B, C, H, W]
    # src_K: [B, 3, 3]
    # src_T: [B, 3, 4]
    # depth: b ,c, h , w
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    if len(src_depmap.shape) == 3:
        src_depmap = src_depmap[:, :, :, None]
    if len(src_depmap.shape) == 4:
        src_depmap = src_depmap.permute(0, 2, 3, 1)

    batch, channels = src_depmap.shape[0], src_depmap.shape[1]
    height, width = src_depmap.shape[2], src_depmap.shape[3]

    with torch.no_grad():
        proj = torch.matmul(ref_proj_new, torch.inverse(src_proj_new))
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_depmap.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_depmap.device)])

        y, x = y.repeat(batch, 1, 1).contiguous(), x.repeat(batch, 1, 1).contiguous()
        xyz = torch.stack((x, y, torch.ones_like(x)), dim=-1) * src_depmap

        # xyz = torch.matmul(torch.linalg.pinv(src_K)[:,None,None,:,:], xyz[:,:,:,:,None])

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz.reshape(150*200,3))
        # o3d.io.write_point_cloud('test.ply',pcd)
        #

        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        rot_xyz = torch.matmul(rot[:, None, None, :, :], xyz[:, :, :, :, None])
        proj_xyz = rot_xyz + trans[:, None, None, :, :]

        proj_xy = proj_xyz[:, :, :, :2, 0] / proj_xyz[:, :, :, 2, 0, None]  # [b,h,w,2(x,y)]

    return proj_xy


if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets.dtu import MVSDataset
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    # mpl.use('Agg')
    import matplotlib.pyplot as plt


    num_depth = 48
    dataset = MVSDataset("/mnt/data4/yswangdata4/dataset/DTU/datasetPM", '/mnt/data4/yswangdata4/dataset/DTU/scan_list_train.txt', 'train',
                         3, num_depth, interval_scale=1) # interval_scale=1.06 * 192 / num_depth

    dataloader = DataLoader(dataset, batch_size=1)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]['stage1']   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][1]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])


    ref_depmap = item['depth']['stage1']
    warped_imgs = warp_feat(src_imgs[1],ref_proj_new, src_proj_new,ref_depmap)
    print('test')


    plt.imshow(warped_imgs[0].permute(1,2,0))
    plt.show()

    # warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)
    #
    # ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    # cv2.imwrite('../tmp/ref.png', ref_img_np)
    # cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    #
    # for i in range(warped_imgs.shape[2]):
    #     warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
    #     img_np = warped_img[0].detach().cpu().numpy()
    #     img_np = img_np[:, :, ::-1] * 255
    #
    #     alpha = 0.5
    #     beta = 1 - alpha
    #     gamma = 0
    #     img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
    #     cv2.imwrite('/mnt/data4/yswangdata4/code/pmnet/test/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))