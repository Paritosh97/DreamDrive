import os
import sys
import shutil
import torch
import numpy as np
import argparse
import time

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the dust3r module within the submodules directory
dust3r_path = os.path.join(current_dir, "..", "..", "submodules", "dust3r")

# Add the dust3r path to sys.path if it's not already there
if dust3r_path not in sys.path:
    print(f"Adding dust3r path to sys.path: {dust3r_path}")
    sys.path.insert(0, dust3r_path)

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:512'

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dreamdrive.geometry.dust3r_utils import (
    compute_global_alignment, 
    load_images, 
    storePly, 
    save_colmap_cameras, 
    save_colmap_images,
    save_depth_map,
    save_conf_map,
    save_point_map,
)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    # parser.add_argument("--model_path", type=str, default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--model_path", type=str, default="submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='cosine')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--focal_avg", action="store_true")
    # parser.add_argument("--focal_avg", type=bool, default=True)

    parser.add_argument("--llffhold", type=int, default=2)
    parser.add_argument("--n_views", type=int, default=12)
    parser.add_argument("--img_base_path", type=str, default="/home/workspace/datasets/instantsplat/Tanks/Barn/24_views")

    return parser

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    n_views = args.n_views
    img_base_path = args.img_base_path
    img_folder_path = os.path.join(img_base_path, "images")
    os.makedirs(img_folder_path, exist_ok=True)
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    ##########################################################################################################################################################################################

    train_img_list = sorted(os.listdir(img_folder_path))
    assert len(train_img_list)==n_views, f"Number of images ({len(train_img_list)}) in the folder ({img_folder_path}) is not equal to {n_views}"

    # save_new_path=True: resize and save for new image
    images, ori_size = load_images(img_folder_path, size=512, save_new_path=True)
    print("ori_size", ori_size)

    start_time = time.time()
    ##########################################################################################################################################################################################
    temp_ckpt_path = os.path.join(img_base_path, "temp", "output.pt")
    if os.path.exists(temp_ckpt_path):
        print(f"Loading pre-cached net outputs from {temp_ckpt_path}")
        output = torch.load(temp_ckpt_path)
        print(f"Loading complete!")
    else:
        pairs = make_pairs(images, scene_graph='video', prefilter=None, symmetrize=True)
        # pairs = make_pairs(images, scene_graph='video', prefilter=None, symmetrize=False)
        output = inference(pairs, model, args.device, batch_size=batch_size)
        os.makedirs(os.path.join(img_base_path, "temp"), exist_ok=True)
        torch.save(output, temp_ckpt_path)
    output_colmap_path=img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
    scene = scene.clean_pointcloud()

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d()) # in world frame 
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())
    ##########################################################################################################################################################################################
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")

    depth_maps = to_numpy(scene.get_depthmaps())
    conf_maps = to_numpy(scene.get_conf())
    
    # save for depth, confidence, and point maps
    # if we'd like to recover depth maps to the original size, simply resize should be fine
    # conf_maps is also the same
    # however, for pts3d, it has already been transformed into the world coordiate, no longer in the camera frame
    # resizing means interpolating the coordinates in another frames, which may bring some errors
    save_depth_map(ori_size, depth_maps, img_folder_path, train_img_list)   
    save_conf_map(ori_size, conf_maps, img_folder_path, train_img_list)
    save_point_map(ori_size, pts3d, img_folder_path, train_img_list)

    # save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", np.array(focals.cpu()))
