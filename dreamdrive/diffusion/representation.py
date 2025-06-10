import os
# os.environ['TORCH_HOME'] = ''
# os.environ['HF_HOME'] = ''
# os.environ['TRANSFORMERS_CACHE'] = ''
import numpy as np
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import tempfile
import subprocess
import imageio
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

from dreamdrive.diffusion.feature_extractor import extract_and_save_features

def get_dinov2_featuremap(img_folder):
    """
    Deprecated: This function is not used in the final implementation.
    """
    # Load DINOv2 model
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True, img_size=518)
    model.eval()

    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # transforms = timm.data.create_transform(**data_config, is_training=False)

    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        img = Image.open(img_path)
        W, H = img.size
        input_size = max(W, H) // 14 * 14
        input_size *= 7 # get large features

        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=Image.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():
            # features = model.forward_features(img_tensor)
            final_feat, intermediates = model.forward_intermediates(img_tensor)
            final_feat = F.interpolate(final_feat, size=(H, W), mode='bilinear', align_corners=False)
        dino_featmap = final_feat.squeeze(0).permute(1, 2, 0)

def get_dinov2_featuremap_v2(img_folder, output_folder, N, ori_size):
    input_img_path_list = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')]
    save_path = os.path.join(os.path.dirname(img_folder.rstrip()), "featmaps")
    os.makedirs(save_path, exist_ok=True)
    saved_feat_path_list = []
    for f in os.listdir(img_folder):
        npyfile = os.path.join(save_path, f.replace(".png", "_dino_reg.npy"))
        saved_feat_path_list.append(npyfile)
    extract_and_save_features(
        input_img_path_list = input_img_path_list,
        saved_feat_path_list= saved_feat_path_list,
        img_shape = (560, 1022),
        stride = 7, # 8 for v1, and 7 for v2
        model_type = "dinov2_vitb14_reg",
    )
    input_folder = save_path
    process_data(input_folder, output_folder, N, ori_size, featkey="dino_reg")

def load_npy_files(folder):
    npy_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
    data = {os.path.splitext(os.path.basename(file))[0]: np.load(file) for file in npy_files}
    return data

def pca(data, n_components):
    T, H, W, C = data.shape
    reshaped_data = data.reshape(T * H * W, C)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)
    pca_result_reshaped = pca_result.reshape(T, H, W, n_components)
    return pca_result_reshaped

def create_pca_video(pca_images, output_file, fps=20):
    T, H, W, C = pca_images.shape
    assert C == 3, "The input data should have 3 channels"

    # Normalize the PCA values to the range [0, 1]
    pca_images_normalized = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

    # Convert to uint8 for imageio
    pca_images_normalized = (255 * pca_images_normalized).astype(np.uint8)
    
    # Write to video
    imageio.mimwrite(output_file, pca_images_normalized, fps=fps)

def save_pca_images(pca_images, names, output_pca_img_path):
    T, H, W, C = pca_images.shape
    assert C == 3, "The input data should have 3 channels"

    # Normalize the PCA values to the range [0, 1]
    pca_images_normalized = (pca_images - pca_images.min()) / (pca_images.max() - pca_images.min())

    # Convert to uint8 for imageio
    pca_images_normalized = (255 * pca_images_normalized).astype(np.uint8)
    
    for i, name in enumerate(names):
        output_file = os.path.join(output_pca_img_path, name + "_pca.png")
        image = Image.fromarray(pca_images_normalized[i])
        image.save(output_file)

def resize(data, height=576, width=1024):
    T, H, W, C = data.shape
    resized_data = np.zeros((T, height, width, C))
    for i in range(T):
        resized_data[i] = cv2.resize(data[i], (width, height))
    return resized_data

# Main processing function
def process_data(input_folder, output_folder, N, ori_size, featkey="in_feats_0"):
    os.makedirs(output_folder, exist_ok=True)
    data_dict = load_npy_files(input_folder)
    feat_path = os.path.join(output_folder, "featmaps")
    os.makedirs(feat_path, exist_ok=True)
    feats, names = [], []
    for key in data_dict.keys():
        if featkey not in key:
            continue
        print(f"Processing {key}")
        feat = data_dict[key]
        feats.append(feat)
        names.append(key)

    # Combine the lists, sort by names, and then unzip them
    sorted_pairs = sorted(zip(names, feats))
    names, feats = zip(*sorted_pairs)

    # Convert back to lists if needed
    names = list(names)
    feats = list(feats)

    feats = np.stack(feats)

    feats_N = pca(feats, N)
    feats_3 = pca(feats, 3)
    feats_N = resize(feats_N, ori_size[0], ori_size[1])
    feats_3 = resize(feats_3, ori_size[0], ori_size[1])
    output_video_file = os.path.join(output_folder, featkey + "_pca.mp4")
    create_pca_video(feats_3, output_video_file)
    print(f"Saved {output_video_file}")
    output_pca_img_path = os.path.join(output_folder, "pca_images")
    os.makedirs(output_pca_img_path, exist_ok=True)    
    save_pca_images(feats_3, names, output_pca_img_path)

    for i, name in enumerate(names):
        save_filename = name + f"_pca{N}.npy"
        output_file = os.path.join(feat_path, save_filename)
        print(f"Saving {output_file}")
        np.save(output_file, feats_N[i])


if __name__ == '__main__':
    
    virtual_path = "data/wild/dynamic/scene_0014/25_views"
    
    # pca feature map for diffusion features
    process_data(
        input_folder=os.path.join(virtual_path, "featmaps"), 
        output_folder=virtual_path, 
        N=32, 
        ori_size=(288, 512), 
        featkey="in_feats_0"
    )

    # pca feature map for dino features
    get_dinov2_featuremap_v2(
        img_folder=os.path.join(virtual_path, "images"), 
        output_folder=virtual_path, 
        N=32, 
        ori_size=(288, 512)
    )
