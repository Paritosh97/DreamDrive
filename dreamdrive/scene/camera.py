
import torch
from torch import nn
import numpy as np
import math
from PIL import Image
from dreamdrive.utils.transform import getWorld2View2, getProjectionMatrix

class CameraModel(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, 
                 image_path, colmap_id, uid, 
                 mask_path = None, depth_path = None, conf_path = None, point_path = None, feat_path = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", fid=None
                 ):
        super(CameraModel, self).__init__()

        self.colmap_id = colmap_id # for loading
        self.uid = uid # for training after shuffling
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_name = image_path.split('/')[-1].split('.')[0]
        image = Image.open(image_path)
        self.original_image = self.PILtoTorch(image).clamp(0.0, 1.0).to(self.data_device) # 3, H, W
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if mask_path is not None:
            mask = Image.open(mask_path)
            self.sky_mask =  self.PILtoTorch(mask).clamp(0.0, 1.0).to(self.data_device)
        else:
            self.sky_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # load time related id
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device) if fid is not None else None

        # TODO: hardcode cx and cy as image center????
        self.fx = self.image_width / (2.0 * math.tan(self.FoVx / 2.0))
        self.fy = self.image_height / (2.0 * math.tan(self.FoVy / 2.0))
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.intrinsic_matrix = self.calculate_intrinsic_matrix()

        # add time attribute (normalized to 0-1)
        # introduced by s3 gaussian waymo process, they seem to also normalize the waymo time to 0-1
        self.time = fid

        # add depth and confidence map to each camera
        if depth_path:
            print(f"Loading depth map from {depth_path}")
            self.depth_map = np.load(depth_path) # (H, W) -> metric depth, ranging from 0 - inf
            self.depth_map = torch.Tensor(self.depth_map).unsqueeze(0).to(self.data_device) # to cuda [1, H, W]

        if conf_path:
            print(f"Loading confidence map from {conf_path}")
            self.conf_map = np.load(conf_path) # (H, W) -> confidence map ranging from 0 - inf
            self.conf_map = self.conf_map / (1.0 + self.conf_map) # we assume values are log(values), and it is same as sigmoid to [0, 1]
            self.conf_map = torch.Tensor(self.conf_map).unsqueeze(0).to(self.data_device) # to cuda [1, H, W]

        if point_path:
            print(f"Loading point cloud from {point_path}")
            self.world_point_cloud = np.load(point_path) # (H, W, 3) -> point cloud in world coordinates
            self.world_point_cloud = torch.Tensor(self.world_point_cloud).to(self.data_device) # to cuda [H, W, 3]

        if feat_path:
            print(f"Loading feature map from {feat_path}")
            self.feature_map = np.load(feat_path)
            self.feature_map = torch.Tensor(self.feature_map).to(self.data_device) # [H, W, C]
            self.feature_map = torch.nn.functional.normalize(self.feature_map, dim=2)

    @property
    def get_depth(self):
        return self.depth_map
    
    @property
    def get_conf(self):
        return self.conf_map

    @property
    def get_world_points(self):
        return self.world_point_cloud

    @property
    def get_feature_map(self):
        return self.feature_map

    def calculate_intrinsic_matrix(self):
        return torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ]).float().cuda()

    def PILtoTorch(self, pil_image):
        torch_image = torch.from_numpy(np.array(pil_image)) / 255.0
        if len(torch_image.shape) == 3:
            return torch_image.permute(2, 0, 1)
        else:
            return torch_image.unsqueeze(dim=-1).permute(2, 0, 1)

    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)

    def get_aabb(self):
        # hard code frustum range 0-80m
        cam_frustum_range = [0.001, 80]
        pix_corners = np.array( # load_size : [h, w]
            [[0,0],[0, self.image_height],[self.image_width, self.image_height],[self.image_width,0]]
        )
        c2w = self.world_view_transform.transpose(0, 1).cpu().numpy()
        
        # sanity check whether c2w is correct
        assert (c2w[:3, :3] - self.R).sum() < 1e-6
        assert (c2w[:3, 3] - self.T).sum() < 1e-6
        
        intri = self.intrinsic_matrix.cpu().numpy()
        frustum = []
        for cam_extent in cam_frustum_range:
            # pix_corners to cam_corners
            cam_corners = np.linalg.inv(intri) @ np.concatenate(
                [pix_corners, np.ones((4, 1))], axis=-1
            ).T * cam_extent
            # cam_corners to world_corners
            world_corners = c2w[:3, :3] @ cam_corners + c2w[:3, 3:4]
            # compute frustum
            frustum.append(world_corners)
        frustum = np.stack(frustum, axis=0)
        flatten_frustum = frustum.transpose(0, 2, 1).reshape(-1, 3)
        aabb_min = np.min(flatten_frustum, axis=0)
        aabb_max = np.max(flatten_frustum, axis=0)
        aabb = np.stack([aabb_min, aabb_max], axis=0)
        return aabb