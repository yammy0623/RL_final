#!/bin/bash +e

# Define common arguments
CONFIG="celeba_hq.yml"
DOC="celeba"
TIMESTEPS=20
ETA=0.85
ETAB=1
SIGMA_0=0.05
INPUT="celeba_hq_sr4_sigma_0.05"
INPUT_ROOT="./"

# Training and evaluation for "sr4"
DEG="sr4"
for DEG in sr4 deblur_uni
do
    echo "Starting training and evaluation for ${DEG}..."
    for TARGET_STEPS in 5 10 20; do
        python train.py --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
            --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
            -i $INPUT --target_steps $TARGET_STEPS --input_root $INPUT_ROOT

        python train.py --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
            --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
            -i $INPUT --second_stage --target_steps $TARGET_STEPS --input_root $INPUT_ROOT

        python eval.py --ni --config $CONFIG --doc $DOC --timesteps $TIMESTEPS \
            --eta $ETA --etaB $ETAB --deg $DEG --sigma_0 $SIGMA_0 \
            -i $INPUT --target_steps $TARGET_STEPS --eval_model_name ${DEG}_2agent_A2C_${TARGET_STEPS} --input_root $INPUT_ROOT
    done

    echo "Finished training and evaluation for ${DEG}."
done