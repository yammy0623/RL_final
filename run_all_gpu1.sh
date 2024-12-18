export CUDA_VISIBLE_DEVICES=1
python train.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 5
python eval.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 5 --eval_model_name deblur_uni_baseline_A2C_5
python train.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 10
python eval.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 10 --eval_model_name deblur_uni_baseline_A2C_10
python train.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 20
python eval.py --ni --config celeba_hq.yml --doc celeba --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.05 -i celeba_hq_sr4_sigma_0.05 --target_steps 20 --eval_model_name deblur_uni_baseline_A2C_20