
import os
import torch
import random
import numpy as np
from typing import List
import torchvision
from dreamdrive.scene.gaussian import GaussianModel, HexPlaneGaussianModel
from dreamdrive.scene.camera import CameraModel
from dreamdrive.utils.transform import getNerfppNorm
from dreamdrive.utils.general import searchForMaxIteration, save_heatmap, save_image
from dreamdrive.scene.loader import (
    fetchPly,
    storePly,
    read_extrinsics_text, 
    read_intrinsics_text,
    read_cams,
    read_cams_nvs,
)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians, load_iteration=None, opt=None, shuffle=True, resolution_scales=[1.0], render="train"):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.source_path = args.source_path
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.shuffle = shuffle
        self.resolution_scales = resolution_scales
        self.filter_unconfident_points = True

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # load cameras
        self.readCameras(args.source_path, args.images, args.eval, args, render)

        # load model and gaussians from trained iterations
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            # self.gaussians.load_nets(
            #     os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "nets.pth")
            # )
        # intialize from point cloud
        else:
            if args.use_semantic_features:
                print("Creating gaussians from pointmaps with semantic features")
                self.gaussians.create_from_pointmaps(
                    image_folder=os.path.join(args.source_path, args.images),
                    spatial_lr_scale=self.cameras_extent,
                    feature_key=args.semantic_feature_key,
                    filter=self.filter_unconfident_points,
                )
            else:
                self.gaussians.create_from_pcd(self.pcd, self.cameras_extent)
            self.gaussians.init_RT_seq(self.train_cameras)

    def readCameras(self, path, images, eval, args, render):

        self.train_cameras, self.test_cameras, self.train_poses, self.test_poses = {}, {}, {}, {}

        ##### For initializing test pose using PCD_Registration
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if images == None else images

        if render == "train":
            print("Loading Training Views (with original poses)...")
            cams_unsorted, poses = read_cams(
                cam_extrinsics=cam_extrinsics, 
                cam_intrinsics=cam_intrinsics, 
                images_folder=os.path.join(path, reading_dir), 
                eval=eval,
            )
        else:
            print(f"Loading Novel Views ({render}) ...")
            cams_unsorted, poses = read_cams_nvs(
                cam_extrinsics=cam_extrinsics, 
                cam_intrinsics=cam_intrinsics, 
                images_folder=os.path.join(path, reading_dir), 
                model_path=args.model_path,
                viewkey=render,
            )

        sorting_indices = sorted(range(len(cams_unsorted)), key=lambda x: cams_unsorted[x].uid)
        cams = [cams_unsorted[i] for i in sorting_indices]
        sorted_poses = [poses[i] for i in sorting_indices]

        train_cams = cams
        train_poses = sorted_poses

        if eval:
            test_cams = cams
            test_poses = sorted_poses
        else:
            test_cams, test_poses = [], []

        nerf_normalization = getNerfppNorm(train_cams)

        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
        
        self.pcd = pcd

        if not self.loaded_iter:
            with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())

        if self.shuffle:
            random.shuffle(train_cams)  # Multi-res consistent random shuffling
            random.shuffle(test_cams)  # Multi-res consistent random shuffling

        # reset uid
        for i, cam in enumerate(train_cams):
            cam.uid = i
        for i, cam in enumerate(test_cams):
            cam.uid = i

        self.cameras_extent = nerf_normalization["radius"]

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = train_cams
            print('train_camera_num: ', len(self.train_cameras[resolution_scale]))
            self.train_poses[resolution_scale] = train_poses
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = test_cams
            print('test_camera_num: ', len(self.test_cameras[resolution_scale]))
            self.test_poses[resolution_scale] = test_poses

        # print("Computing Dyanmic Mask ...")
        # with torch.no_grad():
        #     self.get_dynamic_map()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # self.gaussians.save_nets(os.path.join(point_cloud_path, "nets.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def get_dynamic_map(self):
        """Potentially deprecated function"""
        assert self.shuffle == False
        train_cams: List[CameraModel] = self.getTrainCameras()
        gt_cam = train_cams[0]
        gt_img = gt_cam.original_image # [3, H, W] values ranging from 0 to 1
        H, W = gt_img.shape[1:]
        gt_w2c = gt_cam.world_view_transform.transpose(0, 1) # [4, 4] world to camera transformation matrix
        dynamic_folder = os.path.join(self.source_path, "dynamic")
        os.makedirs(dynamic_folder, exist_ok=True)
        for i, cam in enumerate(train_cams): #[1:]:
            img = cam.original_image # [3, H, W] values ranging from 0 to 1
            pts = cam.get_world_points().reshape(-1, 3) # [H, W, 3] world points
            pts_homo = torch.cat([pts, torch.ones((pts.shape[0], 1), device=pts.device)], dim=1) # [N, 4]
            pts_gt_cam = gt_w2c @ pts_homo.transpose(0,1) # [4, N]
            pts_gt_cam = pts_gt_cam[:3, :] / pts_gt_cam[3:, :] # [3, N]
            intrinsics = cam.intrinsic_matrix
            pixels_gt_cam = intrinsics @ pts_gt_cam # [3, N]
            u, v, d = pixels_gt_cam[0], pixels_gt_cam[1], pixels_gt_cam[2]
            u, v = (u/d).long(), (v/d).long()
            dynamic_mask = torch.zeros((H, W), device=pts.device)
            proj_img = torch.zeros((3, H, W), device=pts.device)
            inside_mask = (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1) & (d > 0) # [N]
            u, v = u[inside_mask], v[inside_mask]
            proj_color = img.reshape(3, -1)[:, inside_mask] # [3, N']
            gt_color = gt_img[:, v, u] # [3, N']
            img_l1 = torch.abs(proj_color - gt_color).mean(0) # [N']
            
            dynamic_mask[v, u] = img_l1
            if i == 0:
                gt_dynamic_mask = dynamic_mask
            else:
                dynamic_mask = dynamic_mask - gt_dynamic_mask
            dynamic_mask_path = os.path.join(dynamic_folder, cam.image_name + "_dynamic.png")

            proj_img[:, v, u] = proj_color
            proj_img_path = os.path.join(dynamic_folder, cam.image_name + "_proj.png")
            
            save_heatmap(dynamic_mask, dynamic_mask_path)
            save_image(proj_img, proj_img_path)
        return 

class HexPlaneScene(Scene):

    gaussians : HexPlaneGaussianModel

    def __init__(self, args, gaussians, load_iteration=None, opt=None, shuffle=True, resolution_scales=[1.0], render="train"):
        super().__init__(args, gaussians, load_iteration, opt, shuffle, resolution_scales, render)

        # load deformation field
        if self.loaded_iter:
            self.gaussians.load_model(os.path.join(self.model_path, "point_cloud/iteration_" + str(self.loaded_iter)))

        if self.loaded_iter: # testing
            aabb = np.load(os.path.join(self.model_path, "aabb.npy"))
            print('cam frustum aabb min: ', aabb[0])
            print('cam frustum aabb max: ', aabb[1])
        else: #training
            aabbs = []
            for scale in self.resolution_scales:
                for cam in self.train_cameras[scale]:
                    aabbs.append(cam.get_aabb())
                for cam in self.test_cameras[scale]:
                    aabbs.append(cam.get_aabb())

            aabbs = np.stack(aabbs, axis=0).reshape(-1,3)
            aabb = np.stack([np.min(aabbs, axis=0), np.max(aabbs, axis=0)], axis=0)
            print('cam frustum aabb min: ', aabb[0])
            print('cam frustum aabb max: ', aabb[1])
            np.save(os.path.join(self.model_path, "aabb.npy"), aabb)

        # for deformation field
        if hasattr(self.gaussians, '_deformation'):
            self.gaussians._deformation.deformation_net.set_aabb(aabb[1], aabb[0])


    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
            
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)    