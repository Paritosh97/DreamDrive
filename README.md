# DreamDrive

This is the repo of DreamDrive, a framework for generative 4D scene modeling from street view images

![Demo Video](docs/everywhere.gif).



## Installation

We tested our environment at Python=3.9.19, CUDA=11.7, and Pytorch=2.0.1+cu117.

### Conda Environments

Intall system packages if needed (Optional).

```
apt-get update
yes | apt-get install libglm-dev libx11-6 libglapi-mesa libgl1-mesa-glx libdrm2 libxcb-glx0 libx11-xcb1 libxcb-dri2-0 libxext6 libxfixes3 libxxf86vm1 libxcb-shm0
```

For conda environment, please try:

```
conda create -n dreamdrive python=3.9
cd PATH_TO_THIS_REPO
pip install -r requirements.txt
python setup.py develop
```


### Third-Party Libs

In submodules/, you will find several folders that needs to be built and intsall:

```
submodules/
    diff-gaussian-rasterization/ # Gaussian splatting that supports feature/depth rendering
    dust3r/ # geometry initialization
    segment-anything-2/ # semantic mask generation for clustering. Optionally, you can also use Kmeans
    simple-knn/ # 3D Gaussian Initialization
    vista/ # video generation
```

Note: Please do not use their original repo, as we have a lot of modifications in their code

For installing the thrid party libs:

```
# diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization
python setup.py install
cd ../../

# dust3r
cd submodules/dust3r
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
python setup.py develop
cd croco/models/curope # in dust3r
python setup.py build_ext --inplace
cd ../../

# segment-anything-2
cd submodules/segment-anything-2
pip install -e .
cd checkpoints # in segment-anything-2
./download_ckpts.sh
cd ..
python setup.py develop
cd ../../

# simple-knn
cd submodules/simple-knn
pip install -e .
cd ../../

# vista
cd submodules/vista
pip install -r requirements.txt
pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
python setup.py develop
# Download the pretrained `svd_xt.safetensors` from [Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/svd_xt.safetensors) and place the checkpoint into `ckpts`.
mkdir ckpts
cd ckpts
wget https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors
cd ../../../
```

You can also refer to the readme in their repos for installation.

## Running

### Video Diffusion Prior

Our method currently rely on a video diffusion prior to generate visual references and features from the input.

For open-sourced video models, we empirically found that [Vista](https://github.com/OpenDriveLab/Vista) and [MagicDrive](https://github.com/cure-lab/MagicDrive) works very well.

The first step is to put your image at `submodules/vista/image_folder` as the input image (we already have some in this folder!).

Then, run 

```
python dreamdrive/diffusion/sample.py
```

to generate frames and feature maps for this input image

Note: the default saving path is 'data/benchmark' [here](dreamdrive/diffusion/sample.py#57). Make sure the path exist and modify based on your needs. Some checkpoints, e.g., DINOv2, need to be downloaded if you want to extract feature maps using DINOv2 encoder.

You will have data in the folder 'data/benchmark' like this:

```
data
----benchmark
     ----scene_0000
          ----25_views
               ----images
               ----featmaps
          ...
     ...
```

### 4D Scenes from Generated Videos 

You can simply run

```
bash scripts/run_ours_v2.sh
```
You will get everything, e.g. 3D guassians, novel-view videos, scores, etc. Remember to change the `DATA_ROOT_DIR` in `scripts/run_ours_v2.sh`.


Optionally, if you encoutered a problem showing that clustering labels are missing, you may need to generate SAM masks for clustering:

```
python dreamdrive/utils/sam.py
```

Or you can also disable sam labels and use Kmeans labels by modifying the code [here](dreamdrive/scene/gaussian.py#315)

## Acknowledgement

Our code is built upon multiple useful repos:

https://github.com/graphdeco-inria/gaussian-splatting \
https://github.com/naver/dust3r \
https://github.com/NVlabs/InstantSplat \
https://github.com/cure-lab/MagicDrive \
https://github.com/OpenDriveLab/Vista



