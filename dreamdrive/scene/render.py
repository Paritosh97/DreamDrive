#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

from dreamdrive.scene.gaussian import GaussianModel, HexPlaneGaussianModel
from dreamdrive.scene.camera import CameraModel
from dreamdrive.utils.gs import eval_sh, inverse_sigmoid
from dreamdrive.utils.transform import get_camera_from_tensor, quadmultiply, from_homogenous, to_homogenous

def render(
    viewpoint_camera: CameraModel,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    camera_pose=None,
    scaling_modifier=1.0,
    override_color=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc._xyz.clone()
    gaussians_rot = pc._rotation.clone()

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = gaussians_rot_trans  # pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, featmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
    )

    rendered_image = rendered_image.clamp(0.0, 1.0)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        "opacity": rendered_alpha,
    }

def render_deform(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, deform, noise_scale, static, is_6dof=False,
           scaling_modifier=1.0, override_color=None, feat=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    fid = viewpoint_camera.fid
    if static:
        d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
    else:
        N = pc.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        ast_noise = torch.randn(1, 1, device='cuda').expand(N, -1) * noise_scale
        d_xyz, d_rotation, d_scaling = deform.step(pc.get_xyz.detach(), time_input + ast_noise)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # Diffrence between mean2D and mean2D_densify is that mean2D_densify accumulate the <absolute> gradient from 2D X and Y directions
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, featmap = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        # means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
        extra_attrs = feat
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points, # screenspace_points_densify, fake it to screen space points
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": rendered_depth,
            "feature_map": featmap,
            }

def render_deform_old(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None, feat=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # Diffrence between mean2D and mean2D_densify is that mean2D_densify accumulate the <absolute> gradient from 2D X and Y directions
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, featmap = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        # means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
        extra_attrs = feat
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points, # screenspace_points_densify, fake it to screen space points
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": rendered_depth,
            "feature_map": featmap,
            }

def render_hexplane(viewpoint_camera: CameraModel, pc : HexPlaneGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", return_decomposition=False, return_dx=False, render_feat=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, dx, feat, dshs = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = shs_final.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    if colors_precomp is not None:
        shs_final = None
    
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, featmap = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp, # [N,3]
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3Ds_precomp = cov3D_precomp)
    
    result_dict = {}
    result_dict.update({
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth":rendered_depth})

    # Concatenate the pre-computation colors and CLIP features indices
    # render_feat = True
    if render_feat and "fine" in stage:
        colors_precomp = feat
        shs_final = None
        rendered_image2, _, _, _, _, _ = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp, # [N,3]
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3Ds_precomp = cov3D_precomp)
        
        result_dict.update({"feat": rendered_image2})

    if return_decomposition and dx is not None:
        dx_abs = torch.abs(dx) # [N,3]
        max_values = torch.max(dx_abs, dim=1)[0] # [N]
        thre = torch.mean(max_values)
        
        dynamic_mask = max_values > thre
        
        rendered_image_d, depth_d, _, _, radii_d, _ = rasterizer(
            means3D = means3D_final[dynamic_mask],
            means2D = means2D[dynamic_mask],
            shs = shs_final[dynamic_mask] if shs_final is not None else None,
            colors_precomp = colors_precomp[dynamic_mask] if colors_precomp is not None else None, # [N,3]
            opacities = opacity[dynamic_mask],
            scales = scales_final[dynamic_mask],
            rotations = rotations_final[dynamic_mask],
            cov3Ds_precomp = cov3D_precomp[dynamic_mask] if cov3D_precomp is not None else None)
        
        rendered_image_s, depth_s, _, _, radii_s, _ = rasterizer(
            means3D = means3D_final[~dynamic_mask],
            means2D = means2D[~dynamic_mask],
            shs = shs_final[~dynamic_mask] if shs_final is not None else None,
            colors_precomp = colors_precomp[~dynamic_mask] if colors_precomp is not None else None, # [N,3]
            opacities = opacity[~dynamic_mask],
            scales = scales_final[~dynamic_mask],
            rotations = rotations_final[~dynamic_mask],
            cov3Ds_precomp = cov3D_precomp[~dynamic_mask] if cov3D_precomp is not None else None
            )
        
        result_dict.update({
            "render_d": rendered_image_d,
            "depth_d":depth_d,
            "visibility_filter_d" : radii_d > 0,
            "render_s": rendered_image_s,
            "depth_s":depth_s,
            "visibility_filter_s" : radii_s > 0,
            })
        
    if return_dx and "fine" in stage:
        result_dict.update({"dx": dx})
        result_dict.update({'dshs' : dshs})

    return result_dict

"""
Rendering deformable GS with camera pose optimization
"""
def render_hybrid(
    viewpoint_camera: CameraModel,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    d_xyz: torch.Tensor, 
    d_rotation: torch.Tensor, 
    d_scaling: torch.Tensor,
    camera_pose=None,
    scaling_modifier=1.0,
    override_color=None,
    feat=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc._xyz.clone()
    gaussians_rot = pc._rotation.clone()

    # apply deformation
    gaussians_xyz += d_xyz
    gaussians_rot += d_rotation


    # transform GS from world to camera frame
    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling # apply deformation
        rotations = gaussians_rot_trans  # pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, featmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
        extra_attrs = feat,
    )

    rendered_image = rendered_image.clamp(0.0, 1.0)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        "opacity": rendered_alpha,
        "feature_map": featmap,
    }

"""
Rendering deformable GS with camera pose optimization and opacity adjustment
"""
def render_hybrid_opacity(
    viewpoint_camera: CameraModel,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    d_xyz: torch.Tensor, 
    d_rotation: torch.Tensor, 
    d_scaling: torch.Tensor,
    d_opacity: torch.Tensor,
    camera_pose=None,
    scaling_modifier=1.0,
    override_color=None,
    feat=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc._xyz.clone()
    gaussians_rot = pc._rotation.clone()

    # apply deformation
    gaussians_xyz += d_xyz
    gaussians_rot += d_rotation


    # transform GS from world to camera frame
    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # change opacity deformation
    opacity = pc.opacity_activation(pc._opacity + d_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling # apply deformation
        rotations = gaussians_rot_trans  # pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, featmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
        extra_attrs = feat,
    )

    rendered_image = rendered_image.clamp(0.0, 1.0)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": rendered_depth,
        "opacity": rendered_alpha,
        "feature_map": featmap,
    }