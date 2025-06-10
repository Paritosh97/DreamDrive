from cdfvd import fvd

def run_fvd(scene_name, gaussian_type, nvs_type, model_name="videomae"):
    fake_video_info = f"data/benchmark/{scene_name}/25_views/gaussians/{gaussian_type}/25_views_30000Iter/{nvs_type}/ours_30000/renders"
    real_video_info = f"data/benchmark/{scene_name}/25_views/images_resized"
    print(f"Running FVD on {fake_video_info} and {real_video_info}")
    assert model_name in ["i3d", "videomae"]
    ckpt_path_dict = {
        "i3d": "submodules/checkpoints/i3d_25views_30000.pth",
        "videomae": "submodules/checkpoints/vit_g_hybrid_pt_1200e_ssv2_ft.pth"
    }
    evaluator = fvd.cdfvd(
        model_name, 
        ckpt_path=ckpt_path_dict[model_name],
        n_real='full', 
        n_fake='full',
    )
    evaluator.compute_fake_stats(evaluator.load_videos(
        video_info=fake_video_info, 
        resolution=256, 
        sequence_length=25, 
        sample_every_n_frames=1,
        data_type='image_folder', 
        num_workers=1, 
        batch_size=25
    ))
    evaluator.compute_real_stats(evaluator.load_videos(
        video_info=real_video_info, 
        resolution=256, 
        sequence_length=25, 
        sample_every_n_frames=1,
        data_type='image_folder', 
        num_workers=1, 
        batch_size=25
    ))
    score = evaluator.compute_fvd_from_stats()
    return score

if __name__ == "__main__":
    # Iterate over scenes from 0000 to 0019
    results = []
    
    for scene in range(0, 20):
        for nvs_type in ["stop", "deceleration", "acceleration"]:
            for gaussian_type in ["hexplane_gaussians"]:
                scene_name = f"scene_{scene:04d}"
                output = run_fvd(scene_name, gaussian_type, nvs_type)
                message = f"{gaussian_type} ({nvs_type}) {scene_name}: {output}\n"
                results.append(message)
                print(message)

    # Write results to a text file
    with open("fvd_results.txt", "w") as file:
        file.writelines(results)

    print("FVD results have been written to fvd_results.txt")