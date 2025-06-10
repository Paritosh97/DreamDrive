import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def init():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def apply_mask(image, mask, obj_id=None, random_color=False):
    # Convert the mask to a binary array
    # mask = mask > 0.0
    mask = mask.astype(np.uint8)
    if random_color:
        color = np.random.randint(0, 255, 3)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array(cmap(cmap_idx)[:3]) * 255

    color = color.astype(np.uint8)

    # Create a colored mask image
    mask_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        mask_image[..., i] = mask * color[i]

    # Convert the original image and mask image to PIL
    original_image = image.convert("RGBA")
    mask_image = Image.fromarray(mask_image, "RGB").convert("RGBA")
    
    # Composite the mask image onto the original image
    combined = Image.blend(original_image, mask_image, alpha=0.1)
    
    return combined.convert("RGB")

def compose_masks(image, masks):
    # Convert the mask to a binary array
    # mask = mask > 0.0
    mask_image = np.zeros_like(image, dtype=np.uint8)

    for obj_id, mask in enumerate(masks):
        cmap = plt.get_cmap("tab10")
        cmap_idx = obj_id
        color = np.array(cmap(cmap_idx)[:3]) * 255

        color = color.astype(np.uint8)

        # Create a colored mask image
        for i in range(3):
            mask_image[mask, i] = color[i]

    # Convert the original image and mask image to PIL
    original_image = image.convert("RGBA")
    mask_image = Image.fromarray(mask_image, "RGB").convert("RGBA")
    
    # Composite the mask image onto the original image
    combined = Image.blend(original_image, mask_image, alpha=0.5)
    
    return combined.convert("RGB")

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

def automatic_mask(video_path):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    init()
    
    image_names = [img_name for img_name in os.listdir(video_path)]
    image_names.sort()

    image = Image.open(os.path.join(video_path, image_names[0]))
    image = np.array(image.convert("RGB"))

    sam2_checkpoint = "submodules/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=50,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.95, # 0.92, # 0.95
        stability_score_offset=1.0, # 0.7, # 1.0
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=False,
    )
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())
    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 
    # # import pdb; pdb.set_trace()
    # save_path = os.path.join(os.path.dirname(video_path.rstrip('/')), 'sam_masks')
    # os.makedirs(save_path, exist_ok=True)
    # plt.savefig(os.path.join(save_path, image_names[0]))
    return masks

def video_seg(video_path, masks=None):
    from sam2.build_sam import build_sam2_video_predictor
    init()
    sam2_checkpoint = "submodules/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    image_names = [img_name for img_name in os.listdir(video_path)]
    image_names.sort()

    new_video_path = os.path.join(os.path.dirname(video_path.rstrip('/')), 'sam_inputs')
    os.makedirs(new_video_path, exist_ok=True)
    for idx, img_name in enumerate(image_names):
        img = Image.open(os.path.join(video_path, img_name))
        new_img_name = str(idx) + ".jpg"
        img.save(os.path.join(new_video_path, new_img_name), 'JPEG')

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=new_video_path)
    predictor.reset_state(inference_state)

    for obj_id, mask in enumerate(masks):
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            mask=mask["segmentation"],
        )
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    save_path = os.path.join(os.path.dirname(video_path.rstrip('/')), 'sam_masks')
    os.makedirs(save_path, exist_ok=True)
    # render the segmentation results every few frames

    for out_frame_idx in range(0, len(image_names)):
        img = Image.open(os.path.join(video_path, image_names[out_frame_idx]))
        sam_mask = np.ones((img.size[1], img.size[0]), dtype=np.float32) * -1
        out_masks = []
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = out_mask[0] # convert from (1, H, W) to (H, W)
            out_masks.append(out_mask)
            sam_mask[out_mask] = out_obj_id
        img = compose_masks(img, out_masks)
        img.save(os.path.join(save_path, image_names[out_frame_idx]))
        np.save(os.path.join(save_path, image_names[out_frame_idx].replace('.png', '_sam.npy')), sam_mask)

    os.system(f"rm -r {str(new_video_path)}")

if __name__ == "__main__":
    for i in range(0,20):
        video_path = f"data/benchmark/scene_{i:04}/25_views/images_resized"
        masks = automatic_mask(video_path)
        video_seg(video_path, masks=masks)