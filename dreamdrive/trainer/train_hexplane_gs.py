import os
import copy
import numpy as np
import torch
import json
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
from dreamdrive.scene.render import render, render_deform, render_hexplane
from dreamdrive.trainer.params import ModelParams, PipelineParams, OptimizationParams
from dreamdrive.trainer.train_utils import save_pose, training_report, dynamic_render, dynamic_render_opacity
from dreamdrive.utils.loss import (
    l1_loss, l2_loss, ssim, psnr, 
    get_linear_noise_func, 
    dynamic_l2_loss, dynamic_uncertainty_loss, dynamic_l1_loss,
    sigmoid_bce_loss, cluster_variance_loss, sigmoid_bce_logits_loss,
    cluster_mask, cluster_mask_v2
)
from dreamdrive.utils.general import Timer, save_heatmap, save_image

def training_hexplane_scene(dataset, opt, pipe, saving_iterations, checkpoint,
                         gaussians, scene, stage, train_iter, timer, args):

    first_iter = 0


    gaussians.training_setup(opt)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    log_file_path = scene.model_path + "/training_log.txt"

    train_cams_init = scene.getTrainCameras().copy()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter + first_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    viewpoint_stack = None

    for iteration in range(first_iter, final_iter+1):        
        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))

        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        render_pkg = render_hexplane(viewpoint_cam, gaussians, pipe, background, stage=stage, return_dx=True, render_feat = True if ('fine' in stage and args.feat_head) else False)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth_pred = render_pkg["depth"]
        images.append(image.unsqueeze(0))
        gt_image = viewpoint_cam.original_image.cuda()

        gt_images.append(gt_image.unsqueeze(0))
        radii_list.append(radii.unsqueeze(0))
        visibility_filter_list.append(visibility_filter.unsqueeze(0))
        viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
        # Loss computation
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        loss = Ll1
        
        # dx loss
        if 'fine' in stage and not args.no_dx and opt.lambda_dx !=0:
            dx_abs = torch.abs(render_pkg['dx'])
            dx_loss = torch.mean(dx_abs) * opt.lambda_dx
            loss += dx_loss
        
        # dshs loss
        if 'fine' in stage and not args.no_dshs and opt.lambda_dshs != 0:
            dshs_abs = torch.abs(render_pkg['dshs'])
            dshs_loss = torch.mean(dshs_abs) * opt.lambda_dshs
            loss += dshs_loss

        # depth loss
        if opt.lambda_depth != 0:
            depth_gt = viewpoint_cam.depth_map
            depth_loss = l2_loss(depth_pred.unsqueeze(0), depth_gt)
            loss += opt.lambda_depth * depth_loss

        # tv regularization loss
        if stage == "fine" and dataset.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(dataset.time_smoothness_weight, dataset.l1_time_planes, dataset.plane_tv_weight)
            loss += tv_loss
        
        # ssim loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        
        # feat loss
        if stage == 'fine' and args.feat_head:
            feat = render_pkg['feat'].to('cuda') # [3,640,960]
            gt_feat = viewpoint_cam.feat_map.permute(2,0,1).to('cuda')
            loss_feat = l2_loss(feat, gt_feat) * opt.lambda_feat
            loss += loss_feat
            
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                dynamic_points = 0
                if 'fine' in stage and not args.no_dx:
                    dx_abs = torch.abs(render_pkg['dx']) # [N,3]
                    max_values = torch.max(dx_abs, dim=1)[0] # [N]
                    thre = torch.mean(max_values)                    
                    mask = (max_values > thre)
                    dynamic_points = torch.sum(mask).item()

                print_dict = {
                    "step": f"{iteration}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "dynamic point": f"{dynamic_points}",
                    "point":f"{total_point}",
                    }
                
                progress_bar.set_postfix(print_dict)
                metrics_file = f"{scene.model_path}/logger.json"
                with open(metrics_file, "a") as f:
                    json.dump(print_dict, f)
                    f.write('\n')

                progress_bar.update(10)
            if iteration == final_iter:
                progress_bar.close()

            timer.pause()
            if (iteration in saving_iterations):
                # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
                save_pose(scene.model_path + 'pose' + f"/pose_{iteration}.npy", gaussians.P, train_cams_init)
                viewpoint_stack = scene.getTrainCameras().copy()
                num_views = len(viewpoint_stack)
                test_psnr = 0
                for viewpoint_cam in viewpoint_stack:
                    render_pkg = render_hexplane(viewpoint_cam, gaussians, pipe, background, stage=stage, return_dx=True, render_feat = True if ('fine' in stage and args.feat_head) else False)
                    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    gt_image = viewpoint_cam.original_image.cuda()
                    psnr_ = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    test_psnr += psnr_
                test_psnr /= num_views
                log_message = "\n[ITER {}] Evaluating test: PSNR {}".format(iteration, test_psnr)
                print(log_message)
                if log_file_path:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(log_message + '\n')


            timer.start()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<2000000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 : # and gaussians.get_xyz.shape[0]>200000
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                # if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                #     gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < final_iter+1:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

def training_hexplane(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint):
    tb_writer = prepare_output_and_logger(dataset)
    
    gaussians = HexPlaneGaussianModel(dataset.sh_degree, dataset)
        
    dataset.model_path = args.model_path
    timer = Timer()
    scene = HexPlaneScene(dataset, gaussians, opt=args)
    timer.start()

    train_cams_init = scene.getTrainCameras().copy()
    os.makedirs(scene.model_path + 'pose', exist_ok=True)
    save_pose(scene.model_path + 'pose' + "/pose_org.npy", gaussians.P, train_cams_init)

    training_hexplane_scene(dataset, opt, pipe, saving_iterations, checkpoint,
                            gaussians, scene, "coarse", opt.coarse_iterations, timer, args)

    training_hexplane_scene(dataset, opt, pipe, saving_iterations, checkpoint,
                            gaussians, scene, "fine", opt.iterations, timer, args)


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
    training_hexplane(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, checkpoint=None)


    print("\nTraining complete.")
