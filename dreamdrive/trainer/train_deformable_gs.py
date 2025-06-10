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
from dreamdrive.scene.render import render, render_deform, render_hexplane
from dreamdrive.trainer.params import ModelParams, PipelineParams, OptimizationParams
from dreamdrive.trainer.train_utils import save_pose, dynamic_render, dynamic_render_opacity
from dreamdrive.utils.loss import (
    l1_loss, l2_loss, ssim, psnr, 
    get_linear_noise_func, 
)
from dreamdrive.utils.general import Timer, save_heatmap, save_image


def training_report(tb_writer, iteration, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, load2gpu_on_the_fly=False, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    log_file_path = scene.model_path + "/training_log.txt"

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(len(scene.getTrainCameras()))]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name']=="train":
                        pose = scene.gaussians.get_RT(viewpoint.uid)
                    else:
                        pose = scene.gaussians.get_RT_test(viewpoint.uid)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    psnr_img = psnr(image, gt_image).mean().double()

                    norm_map = torch.abs(gt_image - image).norm(dim=0, p=2).unsqueeze(0) # [1, H, W]
                    norm_map = norm_map.detach() # as target to train dynamic model
                    save_heatmap(norm_map.squeeze(0), scene.model_path + "/training_dynamics/" + f"{viewpoint.image_name}_{iteration}_psnr_{psnr_img:.2f}.png")
                    
                    psnr_test += psnr_img
                psnr_test /= len(config['cameras'])
                log_message = "\n[ITER {}] Evaluating {}: PSNR {}".format(iteration, config['name'], psnr_test)
                print(log_message)
                
                if log_file_path:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(log_message + '\n')

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        return psnr_test

def training_deform(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(use_semantic_feats=False) # disable semantic features for now
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, opt=args, shuffle=False)
    gaussians.training_setup(opt)

    train_cams_init = scene.getTrainCameras().copy()

    os.makedirs(scene.model_path + 'pose', exist_ok=True)
    save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.P, train_cams_init)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    os.makedirs(scene.model_path + 'training_dynamics', exist_ok=True)
    for iteration in range(1, opt.iterations + 1):
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame
        noise_scale = time_interval * smooth_term(iteration)

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # fid = viewpoint_cam.fid
        if iteration < opt.warm_up:
            static = True
        else:
            static = False

        # Render
        render_pkg_re = render_deform(viewpoint_cam, gaussians, pipe, background, deform, noise_scale, static, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # depth regularization
        if viewpoint_cam.depth_map is not None:
            depth_gt = viewpoint_cam.depth_map
            depth_loss = l2_loss(depth.unsqueeze(0), depth_gt)
            loss += opt.lambda_depth * depth_loss
        
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

            # Log and save
            noise_scale = 0 # for evaluation
            render_args = (pipe, background, deform, noise_scale, static, dataset.is_6dof)
            cur_psnr = training_report(False, iteration, loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render_deform, render_args,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
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
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))

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
    training_deform(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    print("\nTraining complete.")
