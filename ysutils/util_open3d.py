import open3d as o3d
import numpy as np
import cv2

def get_cam_frust(scale=1):
    p0 = np.asarray([0, 0, 0]) * scale
    p1 = np.asarray([-2,1.5,2.5])* scale
    p2 = np.asarray([-2,-1.5,2.5])* scale
    p3 = np.asarray([2,1.5,2.5])* scale
    p4 = np.asarray([2,-1.5, 2.5])* scale

    frust_point = np.stack([p0,p1,p2,p3,p4])
    frust = o3d.geometry.LineSet()
    frust_line_idx = [[0,1],[0,2],[0,3],[0,4],
                     [1,2],[1,3],[4,2],[4,3]]

    frust_line_idx = np.asarray(frust_line_idx)
    frust.points = o3d.utility.Vector3dVector(frust_point)
    frust.lines = o3d.utility.Vector2iVector(frust_line_idx)
    frust.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([[255,0,0]]),(frust_line_idx.shape[0],1)))
    return frust

def gen_cam_frust(poses,scale=0.05):
    poses = [np.asarray(p) for p in poses]
    cams = []
    for i in range(len(poses)):
        cam_mod = get_cam_frust(scale)
        cam_mod.paint_uniform_color([1, 0, 0])
        cams.append(cam_mod.transform(poses[i]))
    return cams


class Gen_ray():
    def __init__(self,H,W,K):
        u, v = np.meshgrid(np.arange(0, W), np.arange(0, H))
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        self.rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        self.K = K
    def gen_rays_at(self,pose):
        """
        Generate rays at world space from one camera.
        pose, not extrinsic!!!!!
        """
        rays_v = np.matmul(pose[:3, :3], self.rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)

    def _get_principal_v(self,K,pose):
        u,v =K[0,2],K[1,2]
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        rays_v = np.matmul(pose[:3, :3], rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)
    def get_principal_v(self,pose):
        K = self.K
        u,v =K[0,2],K[1,2]
        p = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
        p = np.matmul(np.linalg.pinv(K), p.transpose())
        rays_v = p / np.linalg.norm(p, ord=2, axis=0, keepdims=True)
        rays_v = np.matmul(pose[:3, :3], rays_v).transpose()
        rays_o = np.tile(pose[:3, 3].reshape(1,3),(rays_v.shape[0],1))
        return np.concatenate((rays_o,rays_v),axis=1)


def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color

class open3d_raycast_scene():
    def __init__(self):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.GR = None
        self.H = None
        self.W = None
        self.K = None
    def regist_mesh(self, mesh: o3d.geometry.TriangleMesh):
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = self.scene.add_triangles(tmesh)
        return mesh_id
    def regist_camera(self,K,H,W):
        self.GR = Gen_ray(H,W,K[:3,:3])
        self.H=H
        self.W=W
        self.K = K
    def ray_cast(self,pose):
        if self.GR is None:
            raise ValueError('Raycasting camera has not been initialized.')
            return None
        rays = self.GR.gen_rays_at(pose)
        rays = rays.astype(np.float32)
        rays_v = rays[:, 3:6]
        original_ray = self.GR.get_principal_v(pose=pose)
        adj_cos = np.dot(rays_v, original_ray[0, 3:6])
        rays_o3d = o3d.core.Tensor(rays,
                                   dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays_o3d)
        norm = ans['primitive_normals'].numpy()
        norm = norm.reshape(self.H, self.W, 3)
        t_img = ans['t_hit'].numpy()
        depth = (t_img * adj_cos).reshape(self.H, self.W)
        mask = np.logical_not(np.logical_or(np.isinf(depth), np.isnan(depth))) * 255
        return {'depth': depth, 'mask': mask,'normal': norm}
