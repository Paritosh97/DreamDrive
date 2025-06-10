import os
import re
import argparse
from PIL import Image
import shutil

def split_and_save_images(input_image, scene_index, frame_index, output_path):
    # Splitting the 2x3 grid image into 6 sub-images
    camera_positions = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT"
    ]
    
    img = Image.open(input_image)
    width, height = img.size
    sub_width = width // 3
    sub_height = height // 2

    for i in range(2):
        for j in range(3):
            box = (j * sub_width, i * sub_height, (j + 1) * sub_width, (i + 1) * sub_height)
            sub_img = img.crop(box)
            camera_index = i * 3 + j
            sub_img_path = os.path.join(output_path, f"{scene_index}_{frame_index}_{camera_positions[camera_index]}.png")
            sub_img.save(sub_img_path)

def get_scene_images(scene_index, input_path, output_path, sampling=1):
    frames = 16  

    output_path = os.path.join(output_path, f"scene_{scene_index}", f"{frames//sampling*6}_views", "images")
    os.makedirs(output_path, exist_ok=True)

    for frame in range(0, frames, sampling):
        # Find the corresponding image file for the current frame
        pattern = re.compile(rf"{scene_index}_{frame}_gen0_\d+")
        for filename in os.listdir(input_path):
            if pattern.match(filename):
                img_path = os.path.join(input_path, filename)
                print(img_path)
                split_and_save_images(img_path, scene_index, frame, output_path)
                break

    print(f"All sub-images for scene {scene_index} have been saved.")

def get_static_images(scene_index, input_path, output_path, sampling=1): 

    output_path = os.path.join(output_path, f"static_{scene_index}", "6_views", "images")
    os.makedirs(output_path, exist_ok=True)

    # Find the corresponding image file for the current frame
    pattern = re.compile(rf"{scene_index}_gen0.png")
    for filename in os.listdir(input_path):
        if pattern.match(filename):
            img_path = os.path.join(input_path, filename)
            print(img_path)
            split_and_save_images(img_path, scene_index, 0, output_path)
            break

def sample_images(input_dir, output_dir, sampling, first=False):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Function to extract the numerical part from filenames
    def extract_number(filename):
        match = re.search(r'(\d+)\.png$', filename)
        return int(match.group(1)) if match else -1

    # Get list of files in input directory, sorted by the numerical part
    file_list = sorted(os.listdir(input_dir), key=extract_number)

    if first:
        # sample first n frames
        sampled_files = file_list[:sampling]
    else:
        # sample intervals to get n frames
        freq = len(file_list) // sampling
        sampled_files = file_list[::freq]

    for file in sampled_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))

    print("Sampled files copied successfully.")

def sample_images_and_feats(input_dir, output_dir, sampling, first=False):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Function to extract the numerical part from filenames
    def extract_number(filename):
        match = re.search(r'(\d+)\.png$', filename)
        return int(match.group(1)) if match else -1

    # Get list of files in input directory, sorted by the numerical part
    file_list = sorted(os.listdir(input_dir), key=extract_number)

    if first:
        # sample first n frames
        sampled_files = file_list[:sampling]
    else:
        # sample intervals to get n frames
        freq = len(file_list) // sampling
        sampled_files = file_list[::freq]

    for file in sampled_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(output_dir, file))

    print("Sampled files copied successfully.")


def main():
    parser = argparse.ArgumentParser(description='Process scene images and save sub-images.')
    parser.add_argument('--scene', type=int, help='The index of the scene to process')
    parser.add_argument('--sampling', type=int, default=27, help='The index of the scene to process')
    parser.add_argument('--input_path', 
        type=str, 
        default='data/wild/dynamic/scene_0002/46_views/images', 
        help='The path where original images are stored'
    )
    parser.add_argument('--output_path', 
        type=str, 
        default='data/wild/dynamic/scene_0002/27_views/images', 
        help='The output path where processed images are stored'
    )
    
    args = parser.parse_args()
    
    # get_scene_images(args.scene, args.input_path, args.output_path, sampling=args.sampling)
    # get_static_images(args.scene, args.input_path, args.output_path, sampling=args.sampling)
    sample_images(args.input_path, args.output_path, args.sampling, first=True)

if __name__ == "__main__":
    main()
