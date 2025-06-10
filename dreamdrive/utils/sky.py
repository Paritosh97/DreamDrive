import os
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import numpy as np
from PIL import Image

def get_sky_mask(img_folder):
    config_file = ''
    checkpoint_file = ''
    
    # Build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    
    # Get the list of image files
    img_names = [os.path.join(img_folder, fname) for fname in os.listdir(img_folder) if fname.endswith('.png')]
    
    save_folder = os.path.join(os.path.dirname(img_folder.rstrip('/')), 'skymask')
    os.makedirs(save_folder, exist_ok=True)

    for img_name in img_names:
        # Test a single image and show the results
        img = mmcv.imread(img_name)
        result = inference_segmentor(model, img)

        # Assuming the 'sky' class is indexed by the number 10 (example index)
        # You will need to verify the actual index in your model's class mapping
        sky_class_index = 10

        # Create a binary mask where 'sky' class is set to 255 (white) and others to 0 (black)
        sky_mask = result[0] == sky_class_index

        sky_mask = sky_mask.astype(np.uint8) * 255
        
        # Save the mask as an image
        sky_mask_image = Image.fromarray(sky_mask)
        
        
        mask_path = os.path.join(save_folder, img_name.split('/')[-1].replace('.png', '_mask.png'))
        sky_mask_image.save(mask_path)
        print(f"Saved mask to {mask_path}")

        # Color the mask with blue (RGBA: 0, 0, 255, 128)
        blue_mask = np.zeros((sky_mask.shape[0], sky_mask.shape[1], 4), dtype=np.uint8)
        blue_mask[:, :, 2] = sky_mask  # Set the blue channel
        blue_mask[:, :, 3] = sky_mask // 2  # Set the alpha channel to half opacity
        # Convert the original image to RGBA
        original_image = Image.open(img_name).convert("RGBA")
        # Convert the blue mask to an RGBA image
        blue_mask_image = Image.fromarray(blue_mask, mode='RGBA')
        # Blend the blue mask with the original image
        blended_image = Image.alpha_composite(original_image, blue_mask_image)
        blended_path = os.path.join(save_folder, img_name.split('/')[-1].replace('.png', '_blend.png'))
        blended_image.save(blended_path)

    return sky_mask

if __name__ == "__main__":
    for i in range(0, 21):
        img_folder = f'~/DreamDrive/data/benchmark/scene_{i:04}/25_views/images_resized'
        print(f"Processing images in {img_folder}")
        get_sky_mask(img_folder)
