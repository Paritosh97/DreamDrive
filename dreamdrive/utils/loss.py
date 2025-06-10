import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def sigmoid_bce_loss(pred, gt):
    return F.binary_cross_entropy(pred, gt)

def sigmoid_bce_logits_loss(pred, gt):
    return F.binary_cross_entropy_with_logits(pred, gt)

def cluster_variance_loss(dynamic_scores: torch.Tensor, cluster_ids: torch.Tensor):
    unique_clusters = torch.unique(cluster_ids)
    variances = []
    
    for cluster in unique_clusters:
        cluster_mask = (cluster_ids == cluster)
        cluster_scores = dynamic_scores[cluster_mask]
        if cluster_scores.shape[0] > 1:  # Only calculate variance if there are multiple points in the cluster
            cluster_variance = cluster_scores.var(unbiased=False)
            variances.append(cluster_variance)
    
    if variances:
        return torch.stack(variances).mean()
    else:
        return torch.tensor(0.0)

def cluster_mask(dynamic_scores: torch.Tensor, cluster_ids: torch.Tensor, th=0.5):
    dynamic_mask = torch.zeros_like(dynamic_scores)
    unique_clusters = torch.unique(cluster_ids)
    
    for cluster in unique_clusters:
        cluster_mask = (cluster_ids == cluster)
        cluster_scores = dynamic_scores[cluster_mask]
        if cluster_scores.mean() > th:
            dynamic_mask[cluster_mask] = 1.0
    return dynamic_mask


def cluster_mask_v2(dynamic_scores: torch.Tensor, cluster_ids: torch.Tensor, th=0.5, ratio=0.5):
    dynamic_mask = torch.zeros_like(dynamic_scores)
    unique_clusters = torch.unique(cluster_ids)
    
    for cluster in unique_clusters:
        cluster_mask = (cluster_ids == cluster)
        cluster_scores = dynamic_scores[cluster_mask]
        cluster_scores[cluster_scores < th] = 0.0
        cluster_scores[cluster_scores >= th] = 1.0
        if cluster_scores.mean() > ratio: # dynamic points ratio above certain threshold
            dynamic_mask[cluster_mask] = 1.0
    return dynamic_mask

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def weighted_l1_loss(network_output, gt):
    error = torch.abs(network_output - gt).mean(dim=0, keepdim=True) # [1, H, W]
    weights = error / (torch.mean(error) + 1e-8)
    weights = weights.detach()
    loss = torch.abs((network_output - gt)) * weights
    return loss.mean()

def masked_l1_loss(network_output, gt, mask):
    return torch.abs((network_output - gt) * mask).mean()

def dynamic_l2_loss(network_output, gt, uncertainty):
    """
    Dynamic L2 loss. The loss is divided by the uncertainty squared.
    https://arxiv.org/pdf/2405.18715
    """
    loss =  torch.square((network_output - gt)) / (2 * uncertainty.detach() ** 2)
    return loss.mean()

def dynamic_l1_loss(network_output, gt, uncertainty):
    """
    Dynamic L1 loss. The loss is divided by the uncertainty squared.
    https://arxiv.org/pdf/2405.18715
    """
    loss =  torch.abs((network_output - gt)) / (2 * uncertainty.detach() ** 2)
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def compute_ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = torch.finfo(torch.float32).eps**2
    sigma1_sq = torch.clamp(sigma1_sq, min=epsilon)
    sigma2_sq = torch.clamp(sigma2_sq, min=epsilon)
    sigma12 = torch.sign(sigma12) * torch.minimum(
        torch.sqrt(sigma1_sq * sigma2_sq), torch.abs(sigma12))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    c = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    s = (sigma12 + C3) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C3)

    c = torch.clamp(c, max=0.98)
    s = torch.clamp(s, max=0.98)

    return l, c, s

def dynamic_uncertainty_loss(network_output, gt, uncertainty, reg_coef, window_size=11):
    """
    Dynamic SSIM loss (D-SSIM). The loss is divided by the uncertainty squared.
    https://arxiv.org/pdf/2405.18715
    """
    channel = network_output.size(-3)
    window = create_window(window_size, channel)

    if network_output.is_cuda:
        window = window.cuda(network_output.get_device())
    window = window.type_as(network_output)
    l, c, s = compute_ssim(network_output, gt, window, window_size, channel)
    
    ssim_loss = (1 - l.detach()) * (1 - c.detach()) * (1 - s.detach())
    # gradient only flows through the uncertainty
    dynamic_ssim_loss = ssim_loss / (2 * uncertainty ** 2) + reg_coef * torch.log(uncertainty)
    return dynamic_ssim_loss.mean()

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()