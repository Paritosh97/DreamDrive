import os
import json
import torch
from random import randint
import sys
import numpy as np
from tqdm import tqdm
from dreamdrive.scene.render import render_hexplane, render_deform, render_deform_old, render_hybrid, render_hybrid_opacity
from dreamdrive.utils.transform import get_camera_from_tensor
from dreamdrive.scene.scene import Scene
from dreamdrive.utils.loss import l1_loss, l2_loss, ssim, psnr, get_linear_noise_func
from dreamdrive.utils.general import save_heatmap

def save_pose(path, quat_pose, train_cams, llffhold=2):
    output_poses=[]
    index_colmap = [cam.colmap_id for cam in train_cams]
    for quat_t in quat_pose:
        w2c = get_camera_from_tensor(quat_t)
        output_poses.append(w2c)
    combined = list(zip(index_colmap, output_poses))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    colmap_poses = [x[1] for x in sorted_combined]

    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)

def dynamic_render(viewpoint_cam, gaussians, 
                   pipe, background, 
                   camera_pose,
                   deform,
                   dynamic_mask, 
                   noise_scale, 
                   featkey=["dynamic_score"],
                   static=False
    ):
    if static:
         d_xyz, d_rotation, d_scaling = 0, 0, 0
    else:
        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        ast_noise = 0 # torch.randn(1, 1, device='cuda').expand(N, -1) * noise_scale
        noisy_time_input = time_input + ast_noise
        # d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        # d_xyz = d_xyz * dynamic_mask
        # d_rotation = d_rotation * dynamic_mask
        # d_scaling = d_scaling * dynamic_mask
        d_xyz = torch.zeros_like(gaussians.get_xyz, device='cuda')
        d_rotation = torch.zeros_like(gaussians.get_rotation, device='cuda')
        d_scaling = torch.zeros_like(gaussians.get_scaling, device='cuda')
        binary_dynamic_mask = (dynamic_mask > 0.5).squeeze(-1) # [N, 1] -> [N]
        delta_xyz, delta_rotation, delta_scaling = deform.step(gaussians.get_xyz.detach()[binary_dynamic_mask], noisy_time_input[binary_dynamic_mask])
        d_xyz[binary_dynamic_mask] = delta_xyz
        d_rotation[binary_dynamic_mask] = delta_rotation
        d_scaling[binary_dynamic_mask] = delta_scaling

    feats = []
    for key in featkey:
        if key == "dynamic_score":
            feats.append(dynamic_mask)
        elif key == "semantic_feature":
            feats.append(gaussians.get_semantic_features)
        elif key == "uncertainty":
            feats.append(gaussians.get_beta)
        else:
            raise NotImplementedError
    if feats != []:
        feats = torch.cat(feats, dim=1) # [N, c]
    else:
        feats = None

    # Render
    render_pkg_re = render_hybrid(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, camera_pose=camera_pose, feat=feats)    
    
    return render_pkg_re

def dynamic_render_opacity(viewpoint_cam, gaussians, 
                   pipe, background, 
                   camera_pose,
                   deform,
                   dynamic_mask, 
                   noise_scale, 
                   featkey=["dynamic_score"],
                   static=False
    ):
    if static:
         d_xyz, d_rotation, d_scaling, d_opacity = 0, 0, 0, 0
    else:
        fid = viewpoint_cam.fid
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        ast_noise = 0 # if eval else torch.randn(1, 1, device='cuda').expand(N, -1) * noise_scale
        noisy_time_input = time_input + ast_noise
        # d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)
        # d_xyz = d_xyz * dynamic_mask
        # d_rotation = d_rotation * dynamic_mask
        # d_scaling = d_scaling * dynamic_mask
        d_xyz = torch.zeros_like(gaussians.get_xyz, device='cuda')
        d_rotation = torch.zeros_like(gaussians.get_rotation, device='cuda')
        d_scaling = torch.zeros_like(gaussians.get_scaling, device='cuda')
        d_opacity = torch.zeros_like(gaussians.get_opacity, device='cuda')
        binary_dynamic_mask = (dynamic_mask > 0.5).squeeze(-1) # [N, 1] -> [N]
        delta_xyz, delta_rotation, delta_scaling, delta_opacity = deform.step(gaussians.get_xyz.detach()[binary_dynamic_mask], noisy_time_input[binary_dynamic_mask])
        d_xyz[binary_dynamic_mask] = delta_xyz
        d_rotation[binary_dynamic_mask] = delta_rotation
        d_scaling[binary_dynamic_mask] = delta_scaling
        d_opacity[binary_dynamic_mask] = delta_opacity

    feats = []
    for key in featkey:
        if key == "dynamic_score":
            feats.append(dynamic_mask)
        elif key == "semantic_feature":
            feats.append(gaussians.get_semantic_features)
        elif key == "uncertainty":
            feats.append(gaussians.get_beta)
        else:
            raise NotImplementedError
    if feats != []:
        feats = torch.cat(feats, dim=1) # [N, c]
    else:
        feats = None

    # Render
    render_pkg_re = render_hybrid_opacity(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, d_opacity, camera_pose=camera_pose, feat=feats)    
    
    return render_pkg_re

def training_report(tb_writer, iteration, loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, load2gpu_on_the_fly=False, is_6dof=False):
    training_dynamic_save_path = scene.model_path + "/training_dynamics/"
    os.makedirs(training_dynamic_save_path, exist_ok=True)

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
                    # TODO: a bit hacky here: we mannually replace the camera pose here
                    renderArgs[2] = pose
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
