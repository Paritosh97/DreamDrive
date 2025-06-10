import subprocess

# Function to run the pytorch_fid command and capture the output
def run_fid(scene_name, gaussian_type, nvs_type):
    command = [
        "python", "-m", "pytorch_fid",
        f"data/benchmark/{scene_name}/25_views/gaussians/{gaussian_type}/25_views_30000Iter/{nvs_type}/ours_30000/renders",
        f"data/benchmark/{scene_name}/25_views/images_resized"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

if __name__ == "__main__":
    # Iterate over scenes from 0000 to 0019
    results = []
    for gaussian_type in ["hexplane_gaussians"]:
        for scene in range(20):
            for nvs_type in ["stop", "deceleration", "acceleration"]:
                scene_name = f"scene_{scene:04d}"
                output = run_fid(scene_name, gaussian_type, nvs_type)
                message = f"{gaussian_type} ({nvs_type}) {scene_name}: {output}"
                results.append(message)
                print(message)

    # Write results to a text file
    with open("fid_results.txt", "w") as file:
        file.writelines(results)

    print("FID results have been written to fid_results.txt")
