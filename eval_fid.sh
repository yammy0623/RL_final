python eval.py --run_id SAC_v1 --save_path model/sample_model/411 --DM_model model/ddpm_ema_cifar10 --target_steps 10 --img_save_path run/sample/SAC_v1
python fid_score.py run/sample/SAC_v1 run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
