import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
import numpy as np
import imageio
from PIL import Image

from dreamdrive.scene.scene import Scene, HexPlaneScene
from dreamdrive.scene.gaussian import GaussianModel, HexPlaneGaussianModel
from dreamdrive.models.deform_model import DeformModel, DeformOpacityModel
from dreamdrive.models.dynamic_model import DynamicModel, DynamicClusterEmbeddingModel, DynamicClusterModel, DynamicPointModel
from dreamdrive.scene.render import render, render_deform, render_hexplane
from dreamdrive.utils.transform import (
    get_tensor_from_camera,
    generate_interpolated_path,
    generate_shift_path,
    pose_visualizer
)
from dreamdrive.trainer.params import (
    ModelParams,
    PipelineParams,
    get_combined_args
)
from dreamdrive.trainer.train_utils import dynamic_render, dynamic_render_opacity
from dreamdrive.utils.general import save_heatmap
from dreamdrive.utils.loss import cluster_mask, cluster_mask_v2
from dreamdrive.utils.fid import run_fid
from dreamdrive.utils.fvd import run_fvd

def invert_pose(pose):
    """Inverts a single 4x4 pose matrix."""
    rotation = pose[:3, :3].T
    translation = -rotation @ pose[:3, 3]
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = rotation
    inv_pose[:3, 3] = translation
    return inv_pose

def calculate_center_and_radius(org_pose):
    # Calculate the mean position of the original poses
    mean_position = np.mean(org_pose[:, :3, 3], axis=0)
    # Calculate the distances from the mean position to each pose
    distances = np.linalg.norm(org_pose[:, :3, 3] - mean_position, axis=1)
    # Estimate the radius using the standard deviation of the distances
    radius = np.std(distances)
    return mean_position, radius

def generate_circular_poses(center, radius, num_views):
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    poses = []
    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1]  # Keep y constant as the upward direction
        z = center[2] + radius * np.sin(angle)
        rotation = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = [x, y, z]
        poses.append(pose)
    return np.array(poses)

def poses_to_world_to_camera(poses):
    """Converts camera-to-world poses to world-to-camera poses."""
    world_to_camera_poses = np.array([invert_pose(pose) for pose in poses])
    return world_to_camera_poses

def save_circular_pose(model_path, iter, n_views):

    n_interp = 300

    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    pose_visualizer(org_pose, ["green" for _ in org_pose], model_path + "pose/poses_optimized.png")

    center, radius = calculate_center_and_radius(org_pose)

    # Generate circular camera-to-world poses
    circular_poses = generate_circular_poses(center, radius, n_interp)

    # Convert to world-to-camera poses
    inter_pose = poses_to_world_to_camera(circular_poses)

    pose_visualizer(inter_pose, ["blue" for _ in inter_pose], model_path + "pose/poses_interpolated.png")
    np.save(model_path + "pose/pose_interpolated.npy", inter_pose)

def save_interpolate_pose(model_path, iter, n_views):

    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    pose_visualizer(org_pose, ["green" for _ in org_pose], model_path + "pose/poses_optimized.png")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.array(all_inter_pose).reshape(-1, 3, 4)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    pose_visualizer(inter_pose, ["blue" for _ in inter_pose], model_path + "pose/poses_interpolated.png")
    np.save(model_path + "pose/pose_interpolated.npy", inter_pose)

def save_org_pose(model_path, iter):
    org_pose = np.load(model_path + f"pose/pose_org.npy")
    pose_visualizer(org_pose, ["green" for _ in org_pose], model_path + "pose/poses_org.png")
    optimized_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    pose_visualizer(optimized_pose, ["green" for _ in optimized_pose], model_path + "pose/poses_optimized.png")
    np.save(model_path + "pose/pose_optimized.npy", optimized_pose)

def save_stop_pose(model_path, iter):
    key = "stop"
    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    num_steps = len(org_pose)
    save_poses = [org_pose[0] for _ in range(num_steps)]
    save_poses = np.stack(save_poses, 0)
    assert save_poses.shape[0] == num_steps
    pose_visualizer(save_poses, ["blue" for _ in save_poses], model_path + f"pose/poses_{key}.png")
    np.save(model_path + f"pose/pose_{key}.npy", save_poses)

def save_deceleration_pose(model_path, iter):
    key = "deceleration"
    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    num_steps = len(org_pose)
    save_poses = []
    for i in range(num_steps-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=3)
        save_poses.append(tmp_inter_pose[0])
        save_poses.append(tmp_inter_pose[1])
    save_poses = save_poses[:num_steps]
    save_poses = np.stack(save_poses, 0)
    assert save_poses.shape[0] == num_steps
    pose_visualizer(save_poses, ["blue" for _ in save_poses], model_path + f"pose/poses_{key}.png")
    np.save(model_path + f"pose/pose_{key}.npy", save_poses)

def save_acceleration_pose(model_path, iter):
    """accleration then stop"""
    key = "acceleration"
    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    num_steps = len(org_pose)
    save_poses = []
    for i in range(0, num_steps, 2):
        save_poses.append(org_pose[i])
    for _ in range(len(save_poses), num_steps):
        save_poses.append(org_pose[i])
    save_poses = np.stack(save_poses, 0)
    assert save_poses.shape[0] == num_steps
    pose_visualizer(save_poses, ["blue" for _ in save_poses], model_path + f"pose/poses_{key}.png")
    np.save(model_path + f"pose/pose_{key}.npy", save_poses)

def save_shift_pose(model_path, iter, direction, offset):
    key = "shift"
    org_pose = np.load(model_path + f"pose/pose_{iter}.npy")
    num_steps = len(org_pose)
    save_poses = generate_shift_path(org_pose, direction, offset)
    assert save_poses.shape[0] == num_steps
    pose_visualizer(save_poses, ["blue" for _ in save_poses], model_path + f"pose/poses_{key}_{offset}.png")
    np.save(model_path + f"pose/pose_{key}_{offset}.npy", save_poses)

def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)
    output_gif_path = output_video_path.replace(".mp4", ".gif")
    #os.system(f"ffmpeg -i {output_video_path} {output_gif_path}")
    frames = [Image.fromarray(image) for image in images]
    frames[0].save(output_gif_path, save_all=True, append_images=frames[1:], duration=1000/fps, loop=0)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )

def render_set_deform(model_path, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):        
        results = render_deform(view, gaussians, pipeline, background, deform, noise_scale=0, static=False)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
    render_key="stop",
):

    # Applying interpolation
    # save_interpolate_pose(dataset.model_path, iteration, args.n_views)
    # save_circular_pose(dataset.model_path, iteration, args.n_views)
    save_stop_pose(dataset.model_path, iteration)
    save_deceleration_pose(dataset.model_path, iteration)
    save_acceleration_pose(dataset.model_path, iteration)
    save_shift_pose(dataset.model_path, iteration, "left", 1)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False, render=render_key)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.deformable:
        with torch.no_grad():
            deform = DeformModel(False) # dataset.use_semantic_features
            deform.load_weights(dataset.model_path)
        render_set_deform(
            dataset.model_path,
            "interp",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            deform,
        )
    else:
        # render interpolated views
        render_set(
            dataset.model_path,
            "interp",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
        )

    if args.get_video:
        image_folder = os.path.join(dataset.model_path, f'interp/ours_{args.iteration}/renders')
        output_video_file = os.path.join(dataset.model_path, f'{args.scene}_{args.n_views}_{render_key}_view.mp4') # mp4 -> gif
        images_to_video(image_folder, output_video_file, fps=5)

def render_sets_hexplane(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):

    # Applying interpolation
    save_interpolate_pose(dataset.model_path, iteration, args.n_views)
    # save_circular_pose(dataset.model_path, iteration, args.n_views)

    with torch.no_grad():
        gaussians = HexPlaneGaussianModel(dataset.sh_degree, dataset)
        scene = HexPlaneScene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    model_path = dataset.model_path
    name = "interp"
    views = scene.getTrainCameras() # actually get the interpolated views

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_hexplane(
            view, gaussians, pipeline, background, stage="fine", return_dx=False, render_feat=False
        )["render"]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )

    if args.get_video:
        image_folder = os.path.join(dataset.model_path, f'interp/ours_{args.iteration}/renders')
        output_video_file = os.path.join(dataset.model_path, f'{args.scene}_{args.n_views}_view.mp4') # mp4 -> gif
        images_to_video(image_folder, output_video_file, fps=30)


def inference(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
    render_key="optimized",
):

    # Applying interpolation
    # save_interpolate_pose(dataset.model_path, iteration, args.n_views)
    # save_circular_pose(dataset.model_path, iteration, args.n_views)
    save_org_pose(dataset.model_path, iteration)
    save_stop_pose(dataset.model_path, iteration)
    save_deceleration_pose(dataset.model_path, iteration)
    save_acceleration_pose(dataset.model_path, iteration)
    save_shift_pose(dataset.model_path, iteration, "left", 1)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False, render=render_key)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        deform = DeformModel(False) # dataset.use_semantic_features
        # deform = DeformOpacityModel(False) # dataset.use_semantic_features
        deform.load_weights(dataset.model_path, iteration=iteration)

        dynamic = DynamicModel(in_ch=32)
        # dynamic = DynamicPointModel(in_ch=32, n_points=gaussians.get_xyz.shape[0])
        dynamic.load_weights(dataset.model_path, iteration=iteration)

        render_func = dynamic_render

        render_path = os.path.join(dataset.model_path, render_key, "ours_{}".format(iteration), "renders")
        makedirs(render_path, exist_ok=True)
        dynamic_path = os.path.join(dataset.model_path, render_key, "ours_{}".format(iteration), "dynamic_mask")
        makedirs(dynamic_path, exist_ok=True) 
        dynamic_render_path = os.path.join(dataset.model_path, render_key, "ours_{}".format(iteration), "dynamic_render")
        makedirs(dynamic_render_path, exist_ok=True) 
        error_path = os.path.join(dataset.model_path, render_key, "ours_{}".format(iteration), "errors")
        makedirs(error_path, exist_ok=True)

        cluster_ids = gaussians.get_semantic_cluster_id # [N, 1]
        """"""
        # dynamic_scores = dynamic.step(gaussians.get_semantic_features) # [N, 1]
        """"""
        dynamic_scores = gaussians.get_dynamic_score
        """"""
        """Note: temporally disbaled because we have saved the dynamic scores directly to the gaussians"""
        # cluster_threshold = dataset.cluster_th
        # dynamic_mask = cluster_mask_v2(dynamic_scores, cluster_ids, th=cluster_threshold, ratio=0.1) # [N, 1]
        """"""
        dynamic_mask = dynamic_scores
        dynamic_mask = dynamic_mask.detach() # non-differentiablse
        print(f"Number of dynamic points: {dynamic_mask.sum().item()}")
        static = False
        featkey = ["dynamic_score"]
        noise_scale = 0

        views = scene.getTrainCameras()
        time_interval = 1 / len(views)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
            render_args = (pipeline, background, camera_pose, deform, dynamic_mask, noise_scale, featkey, static)
            results = render_func(view, gaussians, *render_args)
            rendering = results["render"]
            dynamic_pred = results["feature_map"]
            # dynamic_pred = (dynamic_pred > 0.99).float()
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(
                rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
            )
            error_map = (rendering - gt).abs().mean(dim=0)
            save_heatmap(error_map.squeeze(0).detach(), os.path.join(error_path, "{0:05d}".format(idx) + ".png"))
            save_heatmap(dynamic_pred.squeeze(0).detach(), os.path.join(dynamic_path, "{0:05d}".format(idx) + ".png"))
            dynamic_img = dynamic_pred * rendering
            torchvision.utils.save_image(
                dynamic_img, os.path.join(dynamic_render_path, "{0:05d}".format(idx) + ".png")
            )

    save_key = render_key
    image_folder = os.path.join(dataset.model_path, f'{render_key}/ours_{iteration}/renders')
    output_video_file = os.path.join(dataset.model_path, f'{args.scene}_{save_key}_{args.n_views}_view.mp4') # mp4 -> gif
    images_to_video(image_folder, output_video_file, fps=5)

    # output_dynamic_video_file = os.path.join(dataset.model_path, f'{args.scene}_{save_key}_{args.n_views}_view_dynamic_mask.mp4') # mp4 -> gif
    # images_to_video(dynamic_path, output_dynamic_video_file, fps=5)

    output_dynamic_render_video_file = os.path.join(dataset.model_path, f'{args.scene}_{save_key}_{args.n_views}_view_dynamic_render.mp4') # mp4 -> gif
    images_to_video(dynamic_render_path, output_dynamic_render_video_file, fps=5)

    # output_error_video_file = os.path.join(dataset.model_path, f'{args.scene}_{save_key}_{args.n_views}_view_error.mp4') # mp4 -> gif
    # images_to_video(error_path, output_error_video_file, fps=5)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--deformable", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    # Initialize system state (RNG)
    # safe_state(args.quiet)
    # inference
    # render_sets
    for render_key in ["optimized", "stop", "deceleration", "acceleration", "shift_1"]:
        inference(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args.skip_train,
            args.skip_test,
            args,
            render_key
        )
        model_path = model.extract(args).model_path
        splits = model_path.split("/")
        scene_name = splits[-6]
        gaussian_type = splits[-3]
        if render_key not in ["stop", "deceleration", "acceleration"]:
            continue
        nvs_type = render_key
        output = run_fid(scene_name, gaussian_type, nvs_type)
        fid_message = f"FID: {gaussian_type} ({nvs_type}) {scene_name}: {output}\n"
        print(fid_message)
        output = run_fvd(scene_name, gaussian_type, nvs_type)
        fvd_message = f"FVD: {gaussian_type} ({nvs_type}) {scene_name}: {output}\n"
        print(fvd_message)
        # Write results to a text file
        save_fname = f"fid_fvd_{nvs_type}.txt"
        with open(os.path.join(model_path, save_fname), "w") as file:
            file.writelines([fid_message, fvd_message])        
