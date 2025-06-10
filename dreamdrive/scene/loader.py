import sys
import os
from plyfile import PlyData, PlyElement
import collections
from PIL import Image
import numpy as np
from dreamdrive.scene.camera import CameraModel
from dreamdrive.utils.transform import (
    BasicPointCloud,
    focal2fov,
    qvec2rotmat,
)

cam_info = collections.namedtuple(
    "cam_info", ["id", "model", "width", "height", "params"])
image_info = collections.namedtuple(
    "image_info", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = image_info(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = cam_info(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_cams(cam_extrinsics, cam_intrinsics, images_folder, eval):

    print(f"Source images are loaded from {images_folder}")

    cams = []
    poses=[]
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        if eval:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[1]
            uid = idx+1

        else:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            uid = intr.id

        height = intr.height
        width = intr.width            
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        poses.append(pose)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))

        mask_folder = os.path.join(os.path.dirname(images_folder.rstrip('/')), 'skymask')
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.png', '_mask.png'))
        if not os.path.exists(mask_path):
            mask_path = None

        # compute time id
        fid = (uid - 1) / (num_frames - 1)

        # add depth and confidence map to each camera
        depth_folder = os.path.join(os.path.dirname(images_folder.rstrip('/')), 'depth')
        depth_path = os.path.join(depth_folder, os.path.basename(image_path).replace('.png', '_depth.npy'))
        if not os.path.exists(depth_path):
            depth_path = None
        conf_folder = os.path.join(os.path.dirname(images_folder.rstrip('/')), 'confidence')
        conf_path = os.path.join(conf_folder, os.path.basename(image_path).replace('.png', '_conf.npy'))
        if not os.path.exists(conf_path):
            conf_path = None
        points_folder = os.path.join(os.path.dirname(images_folder.rstrip('/')), 'pointmaps')
        point_path = os.path.join(points_folder, os.path.basename(image_path).replace('.png', '_pts3d.npy'))
        if not os.path.exists(point_path):
            point_path = None

        feat_folder = os.path.join(os.path.dirname(images_folder.rstrip('/')), 'featmaps')
        feat_path = os.path.join(feat_folder, os.path.basename(image_path).replace('.png', '_dino_reg_pca32.npy'))
        if not os.path.exists(feat_path):
            feat_path = None

        cams.append(
            CameraModel(
                R=R, 
                T=T, 
                FoVx=FovX, 
                FoVy=FovY, 
                image_path=image_path, 
                mask_path=mask_path,
                depth_path=depth_path,
                conf_path=conf_path,
                point_path=point_path,
                feat_path=feat_path,
                colmap_id=uid,
                uid=uid,
                fid=fid
            )
        )
    sys.stdout.write('\n')
    return cams, poses

# For novel view synthesis, open when only render novel view
def read_cams_nvs(cam_extrinsics, cam_intrinsics, images_folder, model_path, viewkey):
    
    print(f"Source images are loaded from {images_folder}")

    pose_nvs_path = model_path + f'pose/pose_{viewkey}.npy'
    pose_nvs = np.load(pose_nvs_path)
    num_frames = len(pose_nvs)
    intr = cam_intrinsics[1]

    cams = []
    poses=[]
    for idx, pose_npy in enumerate(pose_nvs):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, pose_nvs.shape[0]))
        sys.stdout.flush()

        extr = pose_npy
        intr = intr
        height = intr.height
        width = intr.width

        uid = idx
        R = extr[:3, :3].transpose()
        T = extr[:3, 3]
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))

        poses.append(pose)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # dummny image path
        images_list = os.listdir(os.path.join(images_folder))
        image_name_0 = images_list[0]
        image_path = os.path.join(images_folder, image_name_0)
        
        mask_folder = images_folder.replace('images', 'skymask')
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.png', '_mask.png'))
        if not os.path.exists(mask_path):
            mask_path = None

        fid = uid / (num_frames - 1)

        cams.append(
            CameraModel(
                R=R, 
                T=T, 
                FoVx=FovX, 
                FoVy=FovY, 
                image_path=image_path, 
                mask_path=mask_path,
                colmap_id=uid,
                uid=uid,
                fid=fid,
            )
        )

    sys.stdout.write('\n')
    return cams, poses

# For interpolated video, open when only render interpolated video
def read_cams_interp(cam_extrinsics, cam_intrinsics, images_folder, model_path):
    
    print(f"Source images are loaded from {images_folder}")

    pose_interpolated_path = model_path + 'pose/pose_interpolated.npy'
    pose_interpolated = np.load(pose_interpolated_path)
    num_frames = len(pose_interpolated)
    intr = cam_intrinsics[1]

    cams = []
    poses=[]
    for idx, pose_npy in enumerate(pose_interpolated):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, pose_interpolated.shape[0]))
        sys.stdout.flush()

        extr = pose_npy
        intr = intr
        height = intr.height
        width = intr.width

        uid = idx
        R = extr[:3, :3].transpose()
        T = extr[:3, 3]
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))

        poses.append(pose)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # dummny image path
        images_list = os.listdir(os.path.join(images_folder))
        image_name_0 = images_list[0]
        image_path = os.path.join(images_folder, image_name_0)
        
        mask_folder = images_folder.replace('images', 'skymask')
        mask_path = os.path.join(mask_folder, os.path.basename(image_path).replace('.png', '_mask.png'))
        if not os.path.exists(mask_path):
            mask_path = None

        fid = uid / (num_frames - 1)

        cams.append(
            CameraModel(
                R=R, 
                T=T, 
                FoVx=FovX, 
                FoVy=FovY, 
                image_path=image_path, 
                mask_path=mask_path,
                colmap_id=uid,
                uid=uid,
                fid=fid,
            )
        )

    sys.stdout.write('\n')
    return cams, poses