import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read extrinsics data from file
def read_extrinsics_file(file_path, npy_file_path):
    extrinsics = []
    poses = np.load(npy_file_path)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            # import pdb; pdb.set_trace()
            index = int(parts[0])
            rotation = poses[index-1][:3, :3]
            translation = poses[index-1][:3, 3]
            camera_position = np.einsum(
                "...ij,...j->...i", np.linalg.inv(rotation), -translation
            )
            image_name = parts[9]
            extrinsics.append((index, rotation, translation, image_name, camera_position))
    return extrinsics

# Project a 3D point to 2D using camera intrinsics
def project_to_image_plane(point_3d, fx, fy, cx, cy):
    print(point_3d)
    x = (fx * (point_3d[0] / (point_3d[2] + 1e-12))) + cx
    y = (fy * (point_3d[1] / (point_3d[2] + 1e-12))) + cy
    print(x, y)
    return np.array([x, y])

# Base path for the project
base_path = 'data/benchmark/scene_0017/25_views'

# File paths and camera intrinsics
file_path = base_path + '/sparse/0/images.txt'
npy_file_path = base_path + '/gaussians/hybrid_gaussians_v3/25_views_30000Iter/pose/pose_optimized.npy'
focal_path = base_path + '/sparse/0/focal.npy'
f = np.load(focal_path)[0]
fx, fy = f*2, f*2
cx, cy = 256.0*2, 144.0*2

# Read the extrinsics from the file
extrinsics = read_extrinsics_file(file_path, npy_file_path)

# Compute camera positions in world frame and project them directly onto the image plane
projected_points = []
world_points = []
new_world_points = []
_, rotation_cam, translation_cam, _, _ = extrinsics[0]


num_steps = len(extrinsics)
down_offset = np.array([0, 0.015, 0])
offset = 0.015
left_offset = [np.array([i, 0, 0]) for i in np.linspace(0, np.sqrt(offset), num_steps)**2]
right_offset = [np.array([-i, 0, 0]) for i in np.linspace(0, np.sqrt(offset), num_steps)**2]

projected_left_traj, projected_right_traj = [], []
left_traj, right_traj = [], []
for i, (index, rotation, translation, image_name, camera_position_world) in enumerate(extrinsics):
    # Compute the camera position in the world frame
    # camera_position_world = -rotation.T @ translation
    world_points.append(camera_position_world)
    print(index)
    camera_position_world = camera_position_world + down_offset
    new_world_points.append(camera_position_world)

    left_traj_point = left_offset[i] + camera_position_world
    right_traj_point = right_offset[i] + camera_position_world
    left_traj.append(left_traj_point)
    right_traj.append(right_traj_point)

    camera_position_world = rotation_cam @ camera_position_world + translation_cam
    # Project the world frame position onto the image plane
    projected_point = project_to_image_plane(camera_position_world, fx, fy, cx, cy)
    projected_points.append(projected_point)

    left_traj_point_cam = rotation_cam @ left_traj_point + translation_cam
    right_traj_point_cam = rotation_cam @ right_traj_point + translation_cam

    projected_left_traj_point = project_to_image_plane(left_traj_point_cam, fx, fy, cx, cy)
    projected_left_traj.append(projected_left_traj_point)

    projected_right_traj_point = project_to_image_plane(right_traj_point_cam, fx, fy, cx, cy)
    projected_right_traj.append(projected_right_traj_point)

projected_points = np.array(projected_points)
world_points = np.array(world_points)
new_world_points = np.array(new_world_points)
left_traj = np.array(left_traj)
right_traj = np.array(right_traj)
    
# Load the first image
image_base_path = base_path + '/images/'
first_image_path = extrinsics[0][3]
first_image_path = image_base_path + first_image_path
image = cv2.imread(first_image_path)

# Blend the projected camera positions onto the image
for point in projected_points:
    cv2.circle(image, (int(point[0]), int(point[1])), 4, (0, 0, 255), -1)
for point in projected_left_traj:
    cv2.circle(image, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
for point in projected_right_traj:
    cv2.circle(image, (int(point[0]), int(point[1])), 4, (255, 0, 0), -1)

# Display the blended image
plt.figure(figsize=(16, 9))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Projected Camera Positions on the First Image Plane')
plt.axis('off')

# Save the blended image using matplotlib
output_image_path = 'planning.png'
plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

# """
ply_path = base_path + '/cluster_nosky.ply'

# Step 1: Load the original point cloud
original_pcd = o3d.io.read_point_cloud(ply_path)

# Step 2: Create the new 3D points with a distinct color
# Example new 3D points (replace with your actual points)
cam_points = world_points
cam_colors = np.array([[1, 0, 0] for _ in range(cam_points.shape[0])])  # Red color for new points

# Create a point cloud for the new points
cam_pcd = o3d.geometry.PointCloud()
cam_pcd.points = o3d.utility.Vector3dVector(cam_points)
cam_pcd.colors = o3d.utility.Vector3dVector(cam_colors)

new_cam_points = new_world_points
new_cam_colors = np.array([[1, 0, 0] for _ in range(new_cam_points.shape[0])])  # Red color for new points

# Create a point cloud for the new points
new_cam_pcd = o3d.geometry.PointCloud()
new_cam_pcd.points = o3d.utility.Vector3dVector(new_cam_points)
new_cam_pcd.colors = o3d.utility.Vector3dVector(new_cam_colors)

left_cam_points = left_traj
left_cam_colors = np.array([[0, 1, 0] for _ in range(left_cam_points.shape[0])])  # Red color for new points

# Create a point cloud for the new points
left_cam_pcd = o3d.geometry.PointCloud()
left_cam_pcd.points = o3d.utility.Vector3dVector(left_cam_points)
left_cam_pcd.colors = o3d.utility.Vector3dVector(left_cam_colors)

right_cam_points = right_traj
right_cam_colors = np.array([[0, 0, 1] for _ in range(right_cam_points.shape[0])])

right_cam_pcd = o3d.geometry.PointCloud()
right_cam_pcd.points = o3d.utility.Vector3dVector(right_cam_points)
right_cam_pcd.colors = o3d.utility.Vector3dVector(right_cam_colors)

traj_pcd = new_cam_pcd + left_cam_pcd + right_cam_pcd

# Step 3: Combine the original and new point clouds into one
combined_pcd = original_pcd + traj_pcd
# Step 4: Save the combined point cloud to a new PLY file
output_ply_path = "combined_point_cloud.ply"
o3d.io.write_point_cloud(output_ply_path, combined_pcd)
# """
