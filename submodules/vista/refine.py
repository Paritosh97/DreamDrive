import argparse
import json
import random

from pytorch_lightning import seed_everything
from torchvision import transforms

import init_proj_path
from sample_utils import *
from vwm.util import append_dims, instantiate_from_config
from vwm.modules.diffusionmodules.sampling import EulerEDMSampler

VERSION2SPECS = {
    "vwm": {
        "config": "configs/inference/vista.yaml",
        "ckpt": "ckpts/vista.safetensors"
    }
}

DATASET2SOURCES = {
    "NUSCENES": {
        "data_root": "data/nuscenes",
        "anno_file": "annos/nuScenes_val.json"
    },
    "IMG": {
        "data_root": "image_folder"
    },
    "REFINE": {
        "data_root": "outputs/refine",
    }
}


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--version",
        type=str,
        default="vwm",
        help="model version"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="REFINE",
        help="dataset name"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs",
        help="directory to save samples"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="free",
        help="action mode for control, such as traj, cmd, steer, goal"
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=1,
        help="number of sampling rounds"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=25,
        help="number of frames for each round"
    )
    parser.add_argument(
        "--n_conds",
        type=int,
        default=1,
        help="number of initial condition frames for the first round"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="random seed for seed_everything"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="target height of the generated video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="target width of the generated video"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="scale of the classifier-free guidance"
    )
    parser.add_argument(
        "--cond_aug",
        type=float,
        default=0.0,
        help="strength of the noise augmentation"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="number of sampling steps"
    )
    parser.add_argument(
        "--rand_gen",
        action="store_false",
        help="whether to generate samples randomly or sequentially"
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="whether to save memory or not"
    )
    return parser


def get_sample(selected_index=0, dataset_name="REFINE", num_frames=25, action_mode="free"):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    assert dataset_name == "REFINE" 
    
    cond_path = os.path.join(dataset_dict["data_root"], str(selected_index), "condition")
    cond_image_list = os.listdir(cond_path)
    cond_num = len(cond_image_list)
    input_path = os.path.join(dataset_dict["data_root"], str(selected_index), "input")
    input_image_list = os.listdir(input_path)
    # input_image_list = sorted(input_image_list, key=lambda x: int(x.split('_')[2].split('.')[0]))
    input_image_list = sorted(input_image_list)
    input_num = num_frames - cond_num
    path_list = list()
    for cond_file in cond_image_list:
        path_list.append(os.path.join(cond_path, cond_file))
    cnt = 0
    while len(input_image_list) > 0:
        input_file = input_image_list.pop(0)
        path_list.append(os.path.join(input_path, input_file))
        cnt += 1
        if cnt == input_num: # when the number of input images is larger than the required number
            break
        if len(input_image_list) == 0 and cnt < input_num: # when the number of input images is smaller than the required number
            for i in range(input_num - cnt):
                path_list.append(os.path.join(input_path, input_file))

    total_length = cond_num
    return path_list, selected_index, total_length, action_dict


def load_img(file_name, target_height=320, target_width=576, device="cuda"):
    if file_name is not None:
        image = Image.open(file_name)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image file {file_name}")
    ori_w, ori_h = image.size
    # print(f"Loaded input image of size ({ori_w}, {ori_h})")

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        right = (ori_w + tmp_w) // 2
        image = image.crop((left, 0, right, ori_h))
    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        bottom = (ori_h + tmp_h) // 2
        image = image.crop((0, top, ori_w, bottom))
    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])(image)
    return image.to(device)


if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()

    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    sample_index = 3
    seed_everything(opt.seed)

    frame_list, sample_index, condition_length, action_dict = get_sample(sample_index,
                                                                        opt.dataset,
                                                                        opt.n_frames,
                                                                        opt.action)

    img_seq = list()
    for each_path in frame_list:
        img = load_img(each_path, opt.height, opt.width)
        img_seq.append(img)
    images = torch.stack(img_seq)

    value_dict = init_embedder_options(unique_keys)
    cond_img = img_seq[0][None]
    value_dict["cond_frames_without_noise"] = cond_img
    value_dict["cond_aug"] = opt.cond_aug
    value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)
    if action_dict is not None:
        for key, value in action_dict.items():
            value_dict[key] = value


    if opt.n_rounds > 1:
        guider = "TrianglePredictionGuider"
    else:
        guider = "VanillaCFG"
    sampler = init_sampling(guider=guider, steps=opt.n_steps, cfg_scale=opt.cfg_scale, num_frames=opt.n_frames)
    uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]

    num_rounds=opt.n_rounds
    num_frames=opt.n_frames
    force_uc_zero_embeddings=uc_keys
    initial_cond_indices=[index for index in range(condition_length)]
    force_uc_zero_embeddings = default(force_uc_zero_embeddings, list())
    precision_scope = autocast
    device="cuda"

    # refine stage
    with torch.no_grad(), precision_scope(device), model.ema_scope("Sampling"):
        c, uc = get_condition(model, value_dict, num_frames, force_uc_zero_embeddings, device)

        load_model(model.first_stage_model)
        z = model.encode_first_stage(images)
        unload_model(model.first_stage_model)

        load_model(model.denoiser)
        load_model(model.model)

        cond_mask = torch.zeros(num_frames).to(device)
        cond_mask[initial_cond_indices] = 1

        # sigma sampler for adding noise to the latent
        sigma_sampler_config = {
            "target": "vwm.modules.diffusionmodules.sigma_sampling.EDMSampling",
            "params": {
                "p_mean": 1.0,
                "p_std": 1.6,
                "num_frames": num_frames
            }
        }
        sigma_sampler = instantiate_from_config(sigma_sampler_config)
        sigmas = sigma_sampler(z.shape[0]).to(z)
        sigmas = sigmas * 0.25 # sigma=0.4 is the best
        sigma_value = sigmas[0].item() # added noise level
        print(f"added noise sigma value: {sigma_value}")

        # initialize denoise parameters
        discretization_config = {
            "target": "vwm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": 0.002,
                "sigma_max": sigma_value, # sigma_value, # replace deafult 700.0 with the maximum sigma value added by noise
                "rho": 7.0
            }
        }
        guider_config = {
            "target": "vwm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {
                "scale": 2.5,
            }
        }
        sampler = EulerEDMSampler(
            num_steps=1,
            discretization_config=discretization_config,
            guider_config=guider_config,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=999.0,
            s_noise=1.0,
            verbose=False
            # gamma = 0.0, no sigma scaling
        )
        ##

        replace_cond_frames = False
        if replace_cond_frames:
            cond_mask = rearrange(cond_mask, "(b t) -> b t", t=num_frames)
            for each_cond_mask in cond_mask:
                assert len(initial_cond_indices[-1]) < num_frames
                weights = [2 ** n for n in range(len(initial_cond_indices))]
                cond_indices = random.choices(initial_cond_indices, weights=weights, k=1)[0]
                if cond_indices:
                    each_cond_mask[cond_indices] = 1
            cond_mask = rearrange(cond_mask, "b t -> (b t)")        

        noise = torch.randn_like(z)

        # if replace_cond_frames:
        #     sigmas_bc = append_dims((1 - cond_mask) * sigmas, z.ndim)
        # else:
        #     sigmas_bc = append_dims(sigmas, z.ndim)

        sigmas_bc = append_dims((1 - cond_mask) * sigmas, z.ndim)
        # add noise
        noised_z = z + noise * sigmas_bc
        noised_z[initial_cond_indices] = z[initial_cond_indices]

        # denoise
        # sample = noised_z
        # sample, _ = model.denoiser(model.model, sample, sigmas, c, cond_mask)
        # """
        def denoiser(x, sigma, cond, cond_mask):
            return model.denoiser(model.model, x, sigma, cond, cond_mask)
        
        sample, feats = sampler(
            denoiser,
            noised_z, # start from noised_z instead of noise
            cond=c,
            uc=uc,
            cond_frame=z,  # cond_frame will be rescaled when calling the sampler
            cond_mask=cond_mask
        )
        # """

        samples_z = torch.zeros((num_frames, *z.shape[1:])).to(device)

        sample[0] = z[0]
        samples_z[:num_frames] = sample

        unload_model(model.model)
        unload_model(model.denoiser)

        load_model(model.first_stage_model)
        samples_x = model.decode_first_stage(samples_z)
        unload_model(model.first_stage_model)

        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)    

        out = (samples, samples_z, images)

    if isinstance(out, (tuple, list)):
        samples, samples_z, inputs = out
        virtual_path = os.path.join(opt.save, "refine", str(sample_index))
        perform_save_locally(virtual_path, samples, "videos", opt.dataset, sample_index)
        perform_save_locally(virtual_path, samples, "images", opt.dataset, sample_index)
    else:
        raise TypeError
