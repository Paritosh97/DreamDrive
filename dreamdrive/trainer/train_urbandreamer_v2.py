import os
import copy
import numpy as np
import torch
from random import randint
import sys
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from time import perf_counter

from dreamdrive.scene.scene import Scene, HexPlaneScene
from dreamdrive.scene.gaussian import GaussianModel, HexPlaneGaussianModel
from dreamdrive.models.deform_model import DeformModel, DeformOpacityModel
from dreamdrive.models.dynamic_model import DynamicModel, DynamicClusterModel, DynamicClusterEmbeddingModel, DynamicPointModel
from dreamdrive.scene.render import render, render_deform, render_hexplane
from dreamdrive.trainer.params import ModelParams, PipelineParams, OptimizationParams
from dreamdrive.trainer.train_utils import save_pose, training_report, dynamic_render, dynamic_render_opacity
from dreamdrive.utils.loss import (
    l1_loss, l2_loss, ssim, psnr, weighted_l1_loss,
    get_linear_noise_func, 
    dynamic_l2_loss, dynamic_uncertainty_loss, dynamic_l1_loss,
    sigmoid_bce_loss, cluster_variance_loss, sigmoid_bce_logits_loss,
    cluster_mask, cluster_mask_v2, masked_l1_loss
)
from dreamdrive.utils.general import Timer, save_heatmap, save_image

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_dreamdrive(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(use_semantic_feats=False) # disable semantic features for now
    # deform = DeformOpacityModel(use_semantic_feats=False) # disable semantic features for now

    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, opt=args, shuffle=False)
    gaussians.training_setup(opt)

    train_cams_init = scene.getTrainCameras().copy()

    os.makedirs(scene.model_path + 'pose', exist_ok=True)
    save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.P, train_cams_init)

    dynamic = DynamicModel(in_ch=32)
    # dynamic = DynamicPointModel(in_ch=32, n_points=gaussians.get_xyz.shape[0]).cuda()
    dynamic.train_setting(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    
    render_func = dynamic_render

    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    """Training static 3D Gaussians"""
    os.makedirs(scene.model_path + 'training_dynamics', exist_ok=True)
    progress_bar = tqdm(range(opt.static_iterations), desc="Static Training Progress")
    for iteration in range(1, opt.static_iterations + 1):
        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if True: # args.optim_pose==False:
            gaussians.P.requires_grad_(True)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        noise_scale = 0 # no noise since we're not optimizing deformation field
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        camera_pose = gaussians.get_RT(viewpoint_cam.uid) # for camera pose optimization

        N = gaussians.get_xyz.shape[0]
        dynamic_mask = torch.zeros((N, 1), device='cuda') # static
        featkey = []
        static = True
        render_pkg_re = render_func(viewpoint_cam, gaussians, pipe, background, camera_pose, deform, dynamic_mask, noise_scale, featkey, static)
        image, viewspace_point_tensor, visibility_filter, radii, depth, feature_map = render_pkg_re["render"], render_pkg_re["viewspace_points"], \
            render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"], render_pkg_re["feature_map"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]
        # Ll1 = l1_loss(image, gt_image)
        Ll1 = weighted_l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # # depth regularization
        # if viewpoint_cam.depth_map is not None:
        #     depth_gt = viewpoint_cam.depth_map
        #     depth_loss = l2_loss(depth.unsqueeze(0), depth_gt)
        #     loss += opt.lambda_depth * depth_loss
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.static_iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            render_args = [pipe, background, camera_pose, deform, dynamic_mask, noise_scale, featkey, static]
            # Log and save
            cur_psnr = training_report(False, iteration, loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render_func, render_args,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                dynamic.save_weights(args.model_path, iteration)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            # Densification
            # if iteration < opt.densify_until_iter:
            #     viewspace_point_tensor_densify = render_pkg_re["viewspace_points"]
            #     gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.static_iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

    """Training dynamic models"""
    os.makedirs(scene.model_path + 'dynamic', exist_ok=True)

    """Compute Dynamic Masks"""
    os.makedirs(scene.model_path + 'dynamic_mask', exist_ok=True)
    cluster_ids = gaussians._semantic_cluster_id # [N, 1]
    dynamic_mask = torch.zeros_like(cluster_ids).float().detach() # non-differentiablse

    # write dynamic masks to all training views
    viewpoint_stack = scene.getTrainCameras().copy()
    for viewpoint_cam in viewpoint_stack:
        camera_pose = gaussians.get_RT(viewpoint_cam.uid) # for camera pose optimization
        static = True
        noise_scale = 0
        featkey = ["dynamic_score"]
        render_pkg_re = render_func(viewpoint_cam, gaussians, pipe, background, 
                                    camera_pose, deform, dynamic_mask, noise_scale,
                                    featkey, static)
        image, viewspace_point_tensor, visibility_filter, radii, depth, feature_map = render_pkg_re["render"], render_pkg_re["viewspace_points"], \
            render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"], render_pkg_re["feature_map"]
        
        gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]
        norm_map = torch.abs(gt_image - image).norm(dim=0, p=2).unsqueeze(0)
        gt_dynamic_mask = norm_map.detach()
        gt_dynamic_mask = (gt_dynamic_mask > dataset.dynamic_th).float()
        save_heatmap(gt_dynamic_mask.detach().squeeze(0), scene.model_path + "/dynamic_mask/" + f"{viewpoint_cam.image_name}_gt_mask.png")
        np.save(scene.model_path + "/dynamic_mask/" + f"{viewpoint_cam.image_name}_gt_dynamic_mask.npy", gt_dynamic_mask.detach().cpu().numpy())
        save_image(image.detach(), scene.model_path + "/dynamic_mask/" + f"{viewpoint_cam.image_name}_render.png")
    gaussians.append_dynamic_scores(
        source_path = scene.source_path, 
        model_path = scene.model_path, 
        image_key = dataset.images, 
        filter=scene.filter_unconfident_points
    )
    cluster_threshold = dataset.cluster_th
    dynamic_mask = cluster_mask_v2(gaussians.get_dynamic_score, cluster_ids, th=cluster_threshold, ratio=0.1) # [N, 1] dyanmic objects if over 10% points in this cluster have error more than 0.3 norm
    dynamic_mask = dynamic_mask.detach() # non-differentiablse
    print(f"number of dynamic points for unprojected 3d gaussians: {dynamic_mask.sum().item()}")
    
    # densify dynamic Gaussians
    selected_pts_mask = (dynamic_mask > 0.5).squeeze(1) # [N]
    num_dups = 3
    gaussians.densify_dynamic_points(selected_pts_mask, N=num_dups)
    added_dynamic_mask = torch.ones((int(selected_pts_mask.sum())*num_dups, 1), device='cuda')
    dynamic_mask = torch.cat([dynamic_mask, added_dynamic_mask], dim=0) # [N', 1]
    dynamic_mask = dynamic_mask.detach()
    assert dynamic_mask.shape[0] == gaussians.get_xyz.shape[0]
    print(f"number of dynamic points after densify: {dynamic_mask.sum().item()}")


    """Note: this step is a bit hacky: we mannually re-set the gaussians dyanamic score to be the dyanmic mask"""
    gaussians._dynamic_score = dynamic_mask

    """raining Hybrid Gaussians"""
    """Reset opcities for training"""
    gaussians.reset_opacity()
    progress_bar = tqdm(range(opt.iterations - opt.static_iterations), desc="Hybrid Training Progress")
    for iteration in range(opt.static_iterations + 1, opt.iterations + 1):
        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if True: # args.optim_pose==False:
            gaussians.P.requires_grad_(True)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        """Note: this step is a bit hacky: since we have densification, we have to manually re-set the dynamic mask to be dynamic scores in gaussians"""
        dynamic_mask = gaussians._dynamic_score
        dynamic_mask = dynamic_mask.detach() # non-differentiablse

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        noise_scale = time_interval * smooth_term(iteration)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        camera_pose = gaussians.get_RT(viewpoint_cam.uid) # for camera pose optimization

        static = False
        featkey = ["dynamic_score", "uncertainty"]
        render_pkg_re = render_func(viewpoint_cam, gaussians, pipe, background, 
                                    camera_pose, deform, dynamic_mask, noise_scale,
                                    featkey, static)
        image, viewspace_point_tensor, visibility_filter, radii, depth, feature_map = render_pkg_re["render"], render_pkg_re["viewspace_points"], \
            render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"], render_pkg_re["feature_map"]

        uncertainty_pred = feature_map[1:2] + 0.1

        # Loss
        gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]
        
        # Ll1 = l1_loss(image, gt_image)
        Ll1 = weighted_l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # # depth regularization
        # if viewpoint_cam.depth_map is not None:
        #     depth_gt = viewpoint_cam.depth_map
        #     depth_loss = l2_loss(depth.unsqueeze(0), depth_gt)
        #     loss += opt.lambda_depth * depth_loss
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            render_args = [pipe, background, camera_pose, deform, dynamic_mask, noise_scale, featkey, static]
            # Log and save
            cur_psnr = training_report(False, iteration, loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render_func, render_args,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                dynamic.save_weights(args.model_path, iteration)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)


    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7_000, 10_000, 15000, 20_000, 25000, 28000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 5000, 10000, 15000, 20000, 25000, 28000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--optim_pose", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_dreamdrive(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    print("\nTraining complete.")