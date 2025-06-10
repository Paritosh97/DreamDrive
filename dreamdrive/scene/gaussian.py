import torch
import numpy as np
from torch import nn
import os
from os import makedirs
from PIL import Image
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2

from dreamdrive.utils.transform import (
    get_tensor_from_camera, 
    quad2rotation, 
    strip_symmetric, 
    scaling_rotation,
    BasicPointCloud
)
from dreamdrive.utils.gs import inverse_sigmoid, RGB2SH
from dreamdrive.utils.loss import get_expon_lr_func, compute_plane_smoothness
from dreamdrive.utils.general import label_to_color, save_image
from dreamdrive.utils.clustering import dbscan_clustering, agg_clustering, kmeans_clustering
from dreamdrive.scene.loader import storePly
from dreamdrive.models.hexplane_model import deform_network

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        def uncertainty_activation(beta):
            return torch.nn.functional.sigmoid(beta)
        self.uncertainty_activation = uncertainty_activation

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._semantic_features = None # added
        self._semantic_cluster_id = None
        self._beta = None # added for uncertainty
        self._dynamic_score = None # added for dynamic decomposition
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # Add skymodel
        # self.skymodel = EnvLight(resolution=1024, device=torch.device("cuda")).cuda()
        # self.affinemodel = AffineTransform(num_frames=30).cuda()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self._semantic_features, # added
            self._semantic_cluster_id, # added
            self._beta, # added for uncertainty
            self._dynamic_score, # added for dynamic decomposition
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        self._semantic_features, # added
        self._semantic_cluster_id, # added
        self._beta, # added for uncertainty
        self._dynamic_score, # added for dynamic decomposition
        opt_dict,
        self.spatial_lr_scale,
        self.P) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    def compute_relative_world_to_camera(self, R1, t1, R2, t2):
        # Create a row of zeros with a one at the end, for homogeneous coordinates
        zero_row = np.array([[0, 0, 0, 1]], dtype=np.float32)

        # Compute the inverse of the first extrinsic matrix
        E1_inv = np.hstack([R1.T, -R1.T @ t1.reshape(-1, 1)])  # Transpose and reshape for correct dimensions
        E1_inv = np.vstack([E1_inv, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the second extrinsic matrix
        E2 = np.hstack([R2, -R2 @ t2.reshape(-1, 1)])  # No need to transpose R2
        E2 = np.vstack([E2, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the relative transformation
        E_rel = E2 @ E1_inv

        return E_rel

    def init_RT_seq(self, cam_list):
        poses =[]
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)

    def get_RT(self, idx):
        pose = self.P[idx]
        return pose

    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_semantic_features(self):
        return self._semantic_features

    @property
    def get_semantic_cluster_id(self):
        return self._semantic_cluster_id

    @property
    def get_beta(self):
        return self.uncertainty_activation(self._beta)

    @property
    def get_dynamic_score(self):
        return self._dynamic_score

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        Loaded from colmap/dust3r colored point cloud, used in initialization
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_pointmaps(self, image_folder, spatial_lr_scale, feature_key, filter=True):
        """
        Loaded from point maps, attached with feature maps, used in initialization
        """
        self.spatial_lr_scale = spatial_lr_scale

        source_folder = os.path.dirname(image_folder.rstrip("/"))
        point_folder = os.path.join(source_folder, "pointmaps")
        feat_folder = os.path.join(source_folder, "featmaps")
        conf_folder = os.path.join(source_folder, "confidence")
        sky_folder  = os.path.join(source_folder, "skymask")
        pca_folder = os.path.join(source_folder, "pca_images")
        sam_folder = os.path.join(source_folder, "sam_masks")

        image_files = sorted(os.listdir(image_folder), key=lambda x: int(x[-8:-4]))
        point_lists, color_lists, feats, conf_masks, sky_masks, pca_imgs, sam_labels = [], [], [], [], [], [], []
        for image_file in image_files:
            basename = os.path.basename(image_file).split(".")[0]
            img = np.array(Image.open(os.path.join(image_folder, image_file))) / 255.0 # normalized to 0-1
            H, W = img.shape[:2]
            point = np.load(os.path.join(point_folder, basename + "_pts3d.npy"))
            assert point.shape[0] == H and point.shape[1] == W, f"Point map shape {point.shape} does not match image shape {H}x{W}"
            conf = np.load(os.path.join(conf_folder, basename + "_conf.npy"))
            assert conf.shape[0] == H and conf.shape[1] == W, f"Confidence map shape {conf.shape} does not match image shape {H}x{W}"
            conf_mask = conf > 1.0 # hard code 1.0 for filtering
            point_lists.append(point.reshape(-1, 3))
            color_lists.append(img.reshape(-1, 3))
            conf_masks.append(conf_mask.reshape(-1))

            feat_name = os.path.join(feat_folder, basename + "_" + feature_key + ".npy")
            if os.path.exists(feat_name):
                feat = np.load(feat_name)
                assert feat.shape[0] == H and feat.shape[1] == W, f"Feature map shape {feat.shape} does not match image shape {H}x{W}"
                feat_dim = feat.shape[2]
                feats.append(feat.reshape(-1, feat_dim))

            sky_name = os.path.join(sky_folder, basename + "_mask.png")      
            if os.path.exists(sky_name):
                skymask = np.array(Image.open(sky_name)) / 255.0
                if skymask.shape[0] != H or skymask.shape[1] != W:
                    import cv2
                    skymask = cv2.resize(skymask, (W, H))
                skymask = skymask > 0.5
                sky_masks.append(skymask.reshape(-1))
            
            pca_img_name = os.path.join(pca_folder, basename + "_in_feats_0_pca.png")
            if os.path.exists(pca_img_name):
                pca_img = np.array(Image.open(pca_img_name)) / 255.0
                pca_imgs.append(pca_img.reshape(-1, 3))
            
            sam_img_name = os.path.join(sam_folder, basename + "_sam.npy")
            if os.path.exists(sam_img_name):
                sam_label = np.load(sam_img_name)
                sam_labels.append(sam_label.reshape(-1))
            #
            # labels = kmeans_clustering(point.reshape(-1, 3), feat.reshape(-1, feat_dim), img.reshape(-1, 3), weights=[1, 0.1, 0.5], n_clusters=50)
            # labels_rgb = label_to_color(labels)
            # labels_rgb = labels_rgb.reshape(H, W, 3)
            # save_image(labels_rgb, os.path.join(source_folder, "cluster_" + basename + ".png"))
            # import pdb; pdb.set_trace()

        # returned points -> [N, 3] colors -> [N, 3] feats -> [N, feat_dim]
        if filter:
            print("Filtering initial point clouds ...")
            points = np.concatenate([p[m] for p, m in zip(point_lists, conf_masks)])
            colors = np.concatenate([c[m] for c, m in zip(color_lists, conf_masks)])
            if len(feats) > 0:
                feats = np.concatenate([f[m] for f, m in zip(feats, conf_masks)])
            if len(sam_labels) > 0:
                sam_labels = np.concatenate([s[m] for s, m in zip(sam_labels, conf_masks)])
        else:
            points = np.concatenate(point_lists)
            colors = np.concatenate(color_lists)
            if len(feats) > 0:
                feats = np.concatenate(feats)
            if len(sam_labels) > 0:
                sam_labels = np.concatenate(sam_labels)

        if len(sky_masks) > 0:
            nonsky_points = np.concatenate([p[~m1&m2] for p, m1, m2 in zip(point_lists, sky_masks, conf_masks)])
            nonsky_colors = np.concatenate([c[~m1&m2] for c, m1, m2 in zip(color_lists, sky_masks, conf_masks)])
            nonsky_colors = (nonsky_colors * 255.0).astype(np.uint8)
            print("Saving nonsky points to ", os.path.join(source_folder, "cluster_nosky.ply"))
            storePly(os.path.join(source_folder, "cluster_nosky.ply"), nonsky_points, nonsky_colors)
            if len(pca_imgs) > 0:
                nonsky_pca_imgs = np.concatenate([p[~m1&m2] for p, m1, m2 in zip(pca_imgs, sky_masks, conf_masks)])
                nonsky_pca_imgs = (nonsky_pca_imgs * 255.0).astype(np.uint8)
                print("Saving nonsky pca points to ", os.path.join(source_folder, "cluster_pca_nosky.ply"))
                storePly(os.path.join(source_folder, "cluster_pca_nosky.ply"), nonsky_points, nonsky_pca_imgs)
            # import pdb; pdb.set_trace()

        if len(feats) > 0:

            if len(sam_labels) > 0:
                print("use SAM labels ...")
                sam_labels[sam_labels==-1] = sam_labels.max() + 1
                labels = sam_labels
            else:    
                # # spatial temporal clustering
                labels = kmeans_clustering(points.reshape(-1, 3), feats.reshape(-1, feat_dim), colors.reshape(-1, 3), weights=[1, 0.5, 0.5], n_clusters=50)
            labels_rgb = label_to_color(labels)
            storePly(os.path.join(source_folder, "cluster.ply"), points, labels_rgb)
            # import pdb; pdb.set_trace()

            # Normalize feats, avoid large values
            # Note: may not be necessary
            feats = torch.tensor(feats).float().cuda()
            feats = torch.nn.functional.normalize(feats, dim=1)

        ####
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if len(feats) > 0:
            print("Loading with semantic features")
            self._semantic_features = feats
            self._semantic_cluster_id = torch.tensor(labels).unsqueeze(-1).long().cuda() # [N, 1] -> torch long tensor
        else:
            self._semantic_features = None
            self._semantic_cluster_id = None
        print("loading uncertainty ...")
        beta = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        self._beta = nn.Parameter(beta.requires_grad_(True))

    def append_dynamic_scores(self, source_path, model_path, image_key, filter):
        """
        This function leverages pixel-aligned dynamic masks to unproject dynamic scores to 3D gaussians
        """
        conf_folder = os.path.join(source_path, "confidence")
        dyanmic_mask_folder = os.path.join(model_path, "dynamic_mask")
        image_folder = os.path.join(source_path, image_key)
        image_files = sorted(os.listdir(image_folder), key=lambda x: int(x[-8:-4]))
        conf_masks, dynamic_masks = [], []
        for image_file in image_files:
            basename = os.path.basename(image_file).split(".")[0]
            conf = np.load(os.path.join(conf_folder, basename + "_conf.npy"))
            conf_mask = conf > 1.0
            conf_masks.append(conf_mask.reshape(-1))
            dynamic_mask = np.load(os.path.join(dyanmic_mask_folder, basename + "_gt_dynamic_mask.npy")) # [1, H, W]
            assert dynamic_mask.shape[1] == conf.shape[0] and dynamic_mask.shape[2] == conf.shape[1], f"Dynamic mask shape {dynamic_mask.shape} does not match confidence shape {conf.shape}"
            dynamic_masks.append(dynamic_mask.reshape(-1))
        if filter:
            print("Filtering dynamic points ...")
            dynamic_masks = np.concatenate([d[m] for d, m in zip(dynamic_masks, conf_masks)])
        else:
            dynamic_masks = np.concatenate(dynamic_masks)
        self._dynamic_score = torch.tensor(dynamic_masks).unsqueeze(-1).float().cuda()
        assert self._dynamic_score.shape[0] == self.get_xyz.shape[0], f"Dynamic score shape {self._dynamic_score.shape} does not match point cloud shape {self.get_xyz.shape}"

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        l_cam = [{'params': [self.P],'lr': training_args.rotation_lr*0.1, "name": "pose"},]
        # l_cam = [{'params': [self.P],'lr': training_args.rotation_lr, "name": "pose"},]

        l.extend(l_cam)

        if self._beta is not None:
            l.extend([{'params': [self._beta], 'lr': training_args.beta_lr, "name": "beta"}])

        # l_env = [
        #     {'params': self.skymodel.parameters(),'lr': training_args.sky_lr, "name": "sky"},
        #     {'params': self.affinemodel.parameters(),'lr': training_args.affine_lr, "name": "affine"},
        # ]

        # l += l_env

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.cam_scheduler_args = get_expon_lr_func(
                                                    lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=1000)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose":
                lr = self.cam_scheduler_args(iteration)
                # print("pose learning rate", iteration, lr)
                param_group['lr'] = lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self._semantic_features is not None:
            for i in range(self._semantic_features.shape[1]):
                l.append('semantic_{}'.format(i))
        if self._semantic_cluster_id is not None:
            for i in range(self._semantic_cluster_id.shape[1]):
                l.append('cluster_{}'.format(i))
        if self._beta is not None:
            for i in range(self._beta.shape[1]):
                l.append('beta_{}'.format(i))
        if self._dynamic_score is not None:
            for i in range(self._dynamic_score.shape[1]):
                l.append('dynamic_score_{}'.format(i))
        return l

    def save_ply(self, path):
        makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attr_list = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self._semantic_features is not None:
            print("Saving with semantic features")
            semantic_features = self._semantic_features.detach().cpu().numpy()
            attr_list.append(semantic_features)
        if self._semantic_cluster_id is not None:
            print("Saving with cluster id")
            semantic_cluster_id = self._semantic_cluster_id.detach().cpu().numpy()
            attr_list.append(semantic_cluster_id)
        if self._beta is not None:
            print("Saving with uncertainty")
            beta = self._beta.detach().cpu().numpy()
            attr_list.append(beta)
        if self._dynamic_score is not None:
            print("Saving with dynamic scores")
            dynamic_score = self._dynamic_score.detach().cpu().numpy()
            attr_list.append(dynamic_score)
        attributes = np.concatenate(attr_list, axis=1)
        
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_nets(self, path):
        torch.save({
            'skymodel_state_dict': self.skymodel.state_dict(),
            'affinemodel_state_dict': self.affinemodel.state_dict()
        }, path)

    def load_nets(self, path):
        checkpoint = torch.load(path)
        self.skymodel.load_state_dict(checkpoint['skymodel_state_dict'])
        self.affinemodel.load_state_dict(checkpoint['affinemodel_state_dict'])
        print

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_original(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity)*0.1)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        Loaded from saved 3D Gaussians, used in inferencing/resume training
        """
        print("loading gaussian model from exists {}".format(path))

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        semantic_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("semantic_")]
        if len(semantic_names) != 0: # has this attibute in point cloud
            print("Loading with semantic features") 
            semantic_names = sorted(semantic_names, key = lambda x: int(x.split('_')[-1]))
            semantic_feats = np.zeros((xyz.shape[0], len(semantic_names)))
            for idx, attr_name in enumerate(semantic_names):
                semantic_feats[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._semantic_features = torch.tensor(semantic_feats, dtype=torch.float, device="cuda")

        cluster_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("cluster_")]
        if len(cluster_names) != 0: # has this attibute in point cloud
            print("Loading with cluster id") 
            assert len(cluster_names) == 1
            cluster_feats = np.zeros((xyz.shape[0], len(cluster_names)))
            for idx, attr_name in enumerate(cluster_names):
                cluster_feats[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._semantic_cluster_id = torch.tensor(cluster_feats, dtype=torch.long, device="cuda")

        beta_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("beta_")]
        if len(beta_names) != 0: # has this attibute in point cloud
            print("Loading with uncertainty beta")
            assert len(beta_names) == 1
            beta_feats = np.zeros((xyz.shape[0], len(beta_names)))
            for idx, attr_name in enumerate(beta_names):
                beta_feats[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._beta = nn.Parameter(torch.tensor(beta_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        dynamic_score_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("dynamic_score")]
        if len(dynamic_score_names) != 0: # has this attibute in point cloud
            print("Loading with dynamic scores")
            assert len(dynamic_score_names) == 1
            dynamic_score_feats = np.zeros((xyz.shape[0], len(dynamic_score_names)))
            for idx, attr_name in enumerate(dynamic_score_names):
                dynamic_score_feats[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._dynamic_score = torch.tensor(dynamic_score_feats, dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # breakpoint()
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ['pose']: # non-prunable parameters
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if "beta" in optimizable_tensors:
            self._beta = optimizable_tensors["beta"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self._semantic_features is not None:
            self._semantic_features = self._semantic_features[valid_points_mask]
        if self._semantic_cluster_id is not None:
            self._semantic_cluster_id = self._semantic_cluster_id[valid_points_mask]
        if self._dynamic_score is not None:
            self._dynamic_score = self._dynamic_score[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        Add new points to the optimizer
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ['pose']: # non-prunable parameters
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, 
                              new_semantic=None, new_cluster_id=None, new_beta=None, new_dynamic_score=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        if new_beta is not None:
            d["beta"] = new_beta

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if new_semantic is not None:
            self._semantic_features = torch.cat((self._semantic_features, new_semantic), dim=0)
        if new_cluster_id is not None:
            self._semantic_cluster_id = torch.cat((self._semantic_cluster_id, new_cluster_id), dim=0)
        if new_beta is not None:
            self._beta = optimizable_tensors["beta"]
        if new_dynamic_score is not None:
            self._dynamic_score = torch.cat((self._dynamic_score, new_dynamic_score), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quad2rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        if self._semantic_features is not None:
            new_semantic = self._semantic_features[selected_pts_mask].repeat(N,1)
        else:
            new_semantic = None

        if self._semantic_cluster_id is not None:
            new_cluster_id = self._semantic_cluster_id[selected_pts_mask].repeat(N,1)
        else:
            new_cluster_id = None

        if self._beta is not None:
            new_beta = self._beta[selected_pts_mask].repeat(N,1)
        else:
            new_beta = None

        if self._dynamic_score is not None:
            new_dynamic_score = self._dynamic_score[selected_pts_mask].repeat(N,1)
        else:
            new_dynamic_score = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic, new_cluster_id, new_beta, new_dynamic_score)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        if self._semantic_features is not None:
            new_semantic = self._semantic_features[selected_pts_mask]
        else:
            new_semantic = None

        if self._semantic_cluster_id is not None:
            new_cluster_id = self._semantic_cluster_id[selected_pts_mask]
        else:
            new_cluster_id = None

        if self._beta is not None:
            new_beta = self._beta[selected_pts_mask]
        else:
            new_beta = None

        if self._dynamic_score is not None:
            new_dynamic_score = self._dynamic_score[selected_pts_mask]
        else:
            new_dynamic_score = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic, new_cluster_id, new_beta, new_dynamic_score)

    def densify_dynamic_points(self, selected_pts_mask, N=2):
        new_xyz = self._xyz[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N,1)
        new_scaling = self._scaling[selected_pts_mask].repeat(N,1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        if self._semantic_features is not None:
            new_semantic = self._semantic_features[selected_pts_mask].repeat(N,1)
        else:
            new_semantic = None

        if self._semantic_cluster_id is not None:
            new_cluster_id = self._semantic_cluster_id[selected_pts_mask].repeat(N,1)
        else:
            new_cluster_id = None

        if self._beta is not None:
            new_beta = self._beta[selected_pts_mask].repeat(N,1)
        else:
            new_beta = None

        if self._dynamic_score is not None:
            new_dynamic_score = self._dynamic_score[selected_pts_mask].repeat(N,1)
        else:
            new_dynamic_score = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic, new_cluster_id, new_beta, new_dynamic_score)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

class HexPlaneGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, args):
        super().__init__(sh_degree)
        self._deformation = deform_network(args).cuda()
        self._deformation_table = torch.empty(0)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        deform_state,
        self._deformation_table,
        
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        super().create_from_pcd(pcd, spatial_lr_scale)
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()), 'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                lr_pos = lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
        return lr_pos

    def compute_deformation(self,time):    
        deform = self._deformation[:,:,:time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz
    
    def load_model(self, path):
        print("loading deformation field from exists {}".format(path))
        
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]),device="cuda"),0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0],3),device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"),map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"),map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(), os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def save_ply_split(self, dynamic_pcd_path, static_pcd_path, dx_list, visibility_filter):
        makedirs(os.path.dirname(dynamic_pcd_path), exist_ok=True)
        
        dynamic_attributes_list = []  # List to store dynamic attributes
        static_attributes_list = []   # List to store static attributes

        dx = dx_list [24]

        # update xyz
        self._xyz = self._xyz + dx
        
        dx_abs = torch.abs(dx) # [N,3]
        max_values = torch.max(dx_abs, dim=1)[0] # [N]
        thre = torch.mean(max_values)
        
        mask = (max_values > thre)
        dynamic_mask = mask 
        dynamic_mask = dynamic_mask.cpu().numpy() if isinstance(dynamic_mask, torch.Tensor) else dynamic_mask
        dynamic_indices = np.where(dynamic_mask)[0]
        static_mask = ~mask
        static_mask = static_mask.cpu().numpy() if isinstance(static_mask, torch.Tensor) else static_mask
        static_indices = np.where(static_mask)[0]
        # Extract dynamic attributes based on the mask
        dynamic_xyz = self._xyz[dynamic_indices].detach().cpu().numpy()
        dynamic_normals = np.zeros_like(dynamic_xyz)  # Assuming normals are zeros for simplicity
        dynamic_f_dc = self._features_dc[dynamic_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        dynamic_f_rest = self._features_rest[dynamic_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        dynamic_opacities = self._opacity[dynamic_indices].detach().cpu().numpy()
        dynamic_scale = self._scaling[dynamic_indices].detach().cpu().numpy()
        dynamic_rotation = self._rotation[dynamic_indices].detach().cpu().numpy()
        # Extract static attributes based on the mask
        static_xyz = self._xyz[static_indices].detach().cpu().numpy()
        static_normals = np.zeros_like(static_xyz)  # Assuming normals are zeros for simplicity
        static_f_dc = self._features_dc[static_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        static_f_rest = self._features_rest[static_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        static_opacities = self._opacity[static_indices].detach().cpu().numpy()
        static_scale = self._scaling[static_indices].detach().cpu().numpy()
        static_rotation = self._rotation[static_indices].detach().cpu().numpy()
        # Append dynamic and static attributes to their respective lists
        dynamic_attributes_list.append((dynamic_xyz, dynamic_normals, dynamic_f_dc, dynamic_f_rest, dynamic_opacities, dynamic_scale, dynamic_rotation))
        static_attributes_list.append((static_xyz, static_normals, static_f_dc, static_f_rest, static_opacities, static_scale, static_rotation))

        # Concatenate dynamic attributes after the loop
        concatenated_dynamic_attributes = [np.concatenate(attr, axis=0) for attr in zip(*dynamic_attributes_list)]

        # Concatenate static attributes after the loop
        concatenated_static_attributes = [np.concatenate(attr, axis=0) for attr in zip(*static_attributes_list)]
  
        # Prepare PlyData for dynamic point cloud
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        dynamic_elements = np.empty(concatenated_dynamic_attributes[0].shape[0], dtype=dtype_full)
        dynamic_attributes = np.concatenate(concatenated_dynamic_attributes, axis=1)
        dynamic_elements[:] = list(map(tuple, dynamic_attributes))
        dynamic_el = PlyElement.describe(dynamic_elements, 'vertex')

        # Write dynamic PlyData to file
        PlyData([dynamic_el]).write(dynamic_pcd_path)

        # Prepare PlyData for static point cloud
        static_elements = np.empty(concatenated_static_attributes[0].shape[0], dtype=dtype_full)
        static_attributes = np.concatenate(concatenated_static_attributes, axis=1)
        static_elements[:] = list(map(tuple, static_attributes))
        static_el = PlyElement.describe(static_elements, 'vertex')

        # Write static PlyData to file
        PlyData([static_el]).write(static_pcd_path)
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table):
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self._deformation_table = torch.cat([self._deformation_table,new_deformation_table],-1)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # breakpoint()
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quad2rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_deformation_table)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, density_threshold=20, displacement_scale=20, model_path=None, iteration=None, stage=None):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_deformation_table)

    @property
    def get_aabb(self):
        return self._deformation.get_aabb
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, density_threshold, displacement_scale, model_path=None, iteration=None, stage=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, density_threshold, displacement_scale, model_path, iteration, stage)
        self.densify_and_split(grads, max_grad, extent)

    def standard_constaint(self):
        
        means3D = self._xyz.detach()
        scales = self._scaling.detach()
        rotations = self._rotation.detach()
        opacity = self._opacity.detach()
        time =  torch.tensor(0).to("cuda").repeat(means3D.shape[0],1)
        means3D_deform, scales_deform, rotations_deform, _ = self._deformation(means3D, scales, rotations, opacity, time)
        position_error = (means3D_deform - means3D)**2
        rotation_error = (rotations_deform - rotations)**2 
        scaling_erorr = (scales_deform - scales)**2
        return position_error.mean() + rotation_error.mean() + scaling_erorr.mean()

    @torch.no_grad()
    def update_deformation_table(self,threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values/100,threshold)
    
    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
    
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total
    
    def _l1_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
    
    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
