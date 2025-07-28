
INPUT_ROOT="/tmp2/ICML2025"
# Define common arguments
CONFIG="imagenet_256.yml"
DOC="imagenet"
TIMESTEPS=20
ETA=0.85
ETAB=1
SIGMA_0=0.05
EXP="/tmp2/ICML2025/ddrm_2agent"
GPU_IDX=0
# Training and evaluation for "sr4"
DEG="sr4"
echo "Starting training and evaluation for ${DEG}..."
export CUDA_VISIBLE_DEVICES=0
for TARGET_STEPS in 5 10 20; do
    IMAGE_FOLDER="/tmp2/ICML2025/ddrm_2agent/${DOC}/${DOC}_${DEG}_sigma_0.05_${TARGET_STEPS}"
    python train.py  --exp $EXP --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
        --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
        -i $IMAGE_FOLDER --target_steps $TARGET_STEPS --input_root $INPUT_ROOT --gpu_idx $GPU_IDX --start_with "y_addnoise"

    python train.py --exp $EXP --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
        --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
        -i $IMAGE_FOLDER --second_stage --target_steps $TARGET_STEPS --input_root $INPUT_ROOT --gpu_idx $GPU_IDX --start_with "y_addnoise"

    # python eval.py  --exp $EXP --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
    #     --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
    #     -i $IMAGE_FOLDER --target_steps $TARGET_STEPS --eval_model_name ${DEG}_2agent_A2C_${TARGET_STEPS} --input_root $INPUT_ROOT --gpu_idx $GPU_IDX --start_with "y_addnoise"
done

echo "Finished training and evaluation for ${DEG}."
