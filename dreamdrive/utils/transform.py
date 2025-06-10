import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, NamedTuple
import scipy
import matplotlib.pyplot as plt

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def quad2rotation(q):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q).cuda()

    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3)).to(q)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs).cuda()

    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)

    quad, T = inputs[:, :4], inputs[:, 4:]
    w2c = torch.eye(4).to(inputs).float()
    w2c[:3, :3] = quad2rotation(quad)
    w2c[:3, 3] = T
    return w2c

def quadmultiply(q1, q2):
    """
    Multiply two quaternions together using quaternion arithmetic
    """
    # Extract scalar and vector parts of the quaternions
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    # Calculate the quaternion product
    result_quaternion = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    return result_quaternion

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def rotation2quad(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix).cuda()

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """

    if not isinstance(RT, torch.Tensor):
        RT = torch.tensor(RT).cuda()

    rot = RT[:3, :3].unsqueeze(0).detach()
    quat = rotation2quad(rot).squeeze()
    tran = RT[:3, 3].detach()

    return torch.cat([quat, tran])

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).
    
    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.
    
    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        # import pdb; pdb.set_trace()
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points) 

def generate_shift_path(poses, direction, offset):

    def poses_to_points(poses, direction, offset):
        """Converts from pose matrices to (position, lookat, up) format."""
        dist = .1
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        left = np.cross(lookat, up)
        pos = pos + offset * left # apply offset
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])
    
    points = poses_to_points(poses, direction, offset)
    return points_to_poses(points) 

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = quad2rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_torch(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = torch.tensor(translate, dtype=torch.float32)
    
    # Initialize the transformation matrix
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R.t()  # Transpose of R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Compute the inverse to get the camera-to-world transformation
    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    
    # Invert again to get the world-to-view transformation
    Rt = torch.linalg.inv(C2W)
    
    return Rt

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def pose_visualizer(camera_poses, colors, save_path="/mnt/data/1.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for pose, color in zip(camera_poses, colors):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        camera_positions = np.einsum(
            "...ij,...j->...i", np.linalg.inv(rotation), -translation
        )

        # Draw the camera position
        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

        # Draw the camera direction
        # camera_direction = np.linalg.inv(rotation)[:, 2]  # Assuming the camera looks along the Z-axis in its local frame
        # ax.quiver(
        #     camera_positions[0],
        #     camera_positions[1],
        #     camera_positions[2],
        #     camera_direction[0],
        #     camera_direction[1],
        #     camera_direction[2],
        #     color=color,
        #     length=0.05,  # Length of the arrow representing the direction
        #     normalize=True
        # )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")

    plt.savefig(save_path)
    plt.close()

    return save_path

def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    Args:
      R: (3, 3) An orthonormal rotation matrix.
      p: (3,) A 3-vector representing an offset.

    Returns:
      X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
    transform = torch.cat([torch.cat([R, p], dim=-1), bottom_row], dim=1)

    return transform

def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
      w: (N, 3) A 3-vector

    Returns:
      W: (N, 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = torch.zeros(w.shape[0], device=w.device)
    w_skew_list = [zeros, -w[:, 2], w[:, 1],
                   w[:, 2], zeros, -w[:, 0],
                   -w[:, 1], w[:, 0], zeros]
    w_skew = torch.stack(w_skew_list, dim=-1).reshape(-1, 3, 3)
    return w_skew

def exp_so3(w: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
      w: (3,) An axis of rotation.
      theta: An angle of rotation.

    Returns:
      R: (3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    W = skew(w)
    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)  # batch matrix multiplication
    R = identity + torch.sin(theta.unsqueeze(-1)) * W + (1.0 - torch.cos(theta.unsqueeze(-1))) * W_sqr
    return R


def exp_se3(S: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
      S: (6,) A screw axis of motion.
      theta: Magnitude of motion.

    Returns:
      a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)
    theta = theta.view(-1, 1, 1)

    p = torch.bmm((theta * identity + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W_sqr),
                  v.unsqueeze(-1))
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a vector to a homogeneous coordinate vector by appending a 1.

    Args:
        v: A tensor representing a vector or batch of vectors.

    Returns:
        A tensor with an additional dimension set to 1.
    """
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a homogeneous coordinate vector to a standard vector by dividing by the last element.

    Args:
        v: A tensor representing a homogeneous coordinate vector or batch of homogeneous coordinate vectors.

    Returns:
        A tensor with the last dimension removed.
    """
    return v[..., :3] / v[..., -1:]

def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3