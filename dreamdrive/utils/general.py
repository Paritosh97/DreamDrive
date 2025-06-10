import os
import time
import imageio
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.paused = False

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        elif self.paused:
            self.start_time = time.time() - self.elapsed
            self.paused = False

    def pause(self):
        if not self.paused:
            self.elapsed = time.time() - self.start_time
            self.paused = True

    def get_elapsed_time(self):
        if self.paused:
            return self.elapsed
        else:
            return time.time() - self.start_time


def save_img_seq_to_video(out_path, img_seq, fps):
    # img_seq: np array
    writer = imageio.get_writer(out_path, fps=fps)
    for img in img_seq:
        writer.append_data(img)
    writer.close()

def get_img_seq_from_folder(folder, include_key=None, exclude_key=None):
    img_seq = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".png"):
            if (include_key is None or include_key in fname) and (exclude_key is None or exclude_key not in fname):
                img_seq.append(imageio.imread(os.path.join(folder, fname)))
    return img_seq

def save_heatmap(tensor, path):
    array = tensor.cpu().numpy()
    # Normalize the array for better color representation
    array = (array - array.min()) / (array.max() - array.min()) * 255
    # array = np.clip(array, 0, 1) * 255
    array = array.astype(np.uint8)
    # Apply a colormap
    colormap = plt.cm.hot
    heatmap = colormap(array)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    # Save the heatmap
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img.save(path)

def save_image(tensor, path):
    if isinstance(tensor, torch.Tensor):
        array = tensor.cpu().numpy()
    else:
        array = np.array(tensor)
    assert array.ndim == 3 and (array.shape[0] == 3 or array.shape[2] == 3)
    if array.shape[0] == 3:
        # Convert from [3, H, W] to [H, W, 3]
        array = np.transpose(array, (1, 2, 0))
    if array.dtype == np.float32 and array.max() <= 1 and array.min() >= 0: # 0-1
        array = (array * 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    image = Image.fromarray(array)
    image.save(path)

def label_to_color(labels):
    """
    Assign a unique color to each label in the input array.
    Input: [N]
    Output: [N, 3] RGB values -> [0, 255]
    """
    unique_labels = np.unique(labels)
    colors = sns.color_palette('hsv', len(unique_labels))
    colors_255 = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    label_color_mapping = {label: colors_255[i] for i, label in enumerate(unique_labels)}
    # Create an array of colors corresponding to the input labels
    color_array = np.array([label_color_mapping[label] for label in labels]) 
    return color_array

if __name__ == "__main__":
    folder = "data/wild/dynamic/scene_0002/27_views/dynamic"
    out_path = folder + "/video.mp4"
    img_seq = get_img_seq_from_folder(folder, include_key="proj", exclude_key=None)
    save_img_seq_to_video(out_path, img_seq, 10)
