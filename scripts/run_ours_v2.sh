#! /bin/bash

GPU_ID=0
DATA_ROOT_DIR="enter the data root directory here"

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=30000
pose_lr=1x

DATASET=benchmark
N_VIEW=25

# Loop over scenes scene_0004 to scene_0007
for SCENE_NUM in {0..19}
do
    SCENE=$(printf "scene_%04d" $SCENE_NUM)
    echo $SCENE

    # SOURCE_PATH must be Absolute path
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views
    MODEL_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/${N_VIEW}_views/gaussians/hybrid_gaussians_v3/${N_VIEW}_views_${gs_train_iter}Iter/

    CHECK_PATH=${MODEL_PATH}/point_cloud/iteration_${gs_train_iter}

    if [ -d "${CHECK_PATH}" ]; then
        echo "Skipping ${SCENE} as ${CHECK_PATH} exists."
        continue
    fi

    # # ----- (1) Dust3r_coarse_geometric_initialization -----
    CMD1="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore dreamdrive/geometry/dust3r_init.py \
    --img_base_path ${SOURCE_PATH} \
    --n_views ${N_VIEW}  \
    --focal_avg \
    "

    # # ----- (1.1) Optional: generating SAM labels -----
    CMDSAM="python dreamdrive/utils/sam.py"

    # # ----- (2) Train: jointly optimize pose -----
    CMD2="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore dreamdrive/trainer/train_dreamdrive_v2.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH}  \
    --n_views ${N_VIEW}  \
    --scene ${SCENE} \
    --iter ${gs_train_iter} \
    --optim_pose \
    "

    # ----- (3) Render interpolated pose & output video -----
    CMD3="CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore dreamdrive/trainer/inference_v2.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH}  \
    --n_views ${N_VIEW}  \
    --scene ${SCENE} \
    --iter ${gs_train_iter} \
    --get_video \
    --deformable \
    --eval \
    "

    echo "========= ${SCENE}: Geometry Initialization ========="
    eval $CMD1
    # echo "========= ${SCENE}: Generating SAM2 labels for Clustering ========="
    # eval $CMDSAM
    echo "========= ${SCENE}: Train: jointly optimize pose ========="
    eval $CMD2
    echo "========= ${SCENE}: Render interpolated pose & output video ========="
    eval $CMD3
done
