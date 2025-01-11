export CUDA_VISIBLE_DEVICES=1
INPUT_ROOT="/tmp2/ICML2025"
EXP="/tmp2/ICML2025/ddrm_rl_prior"
IMAGE_FOLDER="/tmp2/ICML2025/ddrm_rl_prior/image_samples/celeba_hq_sr4_sigma_0.05_2"
python train.py --exp $EXP --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i $IMAGE_FOLDER --target_steps 5 --input_root $INPUT_ROOT
python eval.py --exp $EXP --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i $IMAGE_FOLDER --target_steps 5 --eval_model_name sr4_baseline_A2C_5 --input_root $INPUT_ROOT