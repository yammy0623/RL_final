import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, SubprocVecEnv
from train import CustomCNN
import numpy as np
from collections import Counter
from func import MD_SAC
from tqdm import tqdm
import argparse

register(
    id='final-eval',
    entry_point='envs:EvalDiffusionEnv'
)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="SAC_v1")
    parser.add_argument("--save_path", type=str, default="model/sample_model/262")
    parser.add_argument("--DM_model", type=str, default="model/ddpm_ema_cifar10")
    parser.add_argument("--target_steps", type=int, default=100)
    parser.add_argument("--img_save_path", type=str, default="run/sample/SAC_DDIM_100")
    parser.add_argument("--action_range", type=float, default=1.0, help="Range of the action")
    return parser.parse_args()

def make_env(my_config):
    def _init():
        config = {
            "model_name": my_config["DM_model"],
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
            "img_save_path": my_config["img_save_path"],
            "action_range": my_config["action_range"]
        }
        return gym.make('final-eval', **config)
    return _init

def evaluation(env, model, eval_num=100):
    """We only evaluate seeds 0-99 as our public test cases."""

    ### Run eval_num times rollouts,
    for _ in tqdm(range(eval_num)):
        done = False
        # Set seed and reset env using Gymnasium API
        obs = env.reset()

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    args = arg_parser()
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )
    my_config = {
        "run_id": args.run_id,

        "algorithm": MD_SAC,
        "policy_network": "MultiInputPolicy",
        "save_path": args.save_path,

        "policy_kwargs": policy_kwargs,

        "DM_model": args.DM_model,
        "target_steps": args.target_steps,
        "max_steps": args.target_steps,

        "num_eval_envs": 1,
        "eval_num": 50000,
        "img_save_path": args.img_save_path,
        "action_range": args.action_range
    }
    ### Load model with SB3
    model = SAC.load(my_config['save_path'])
    env = DummyVecEnv([make_env(my_config) for _ in range(my_config['num_eval_envs'])])
    
    evaluation(env, model, my_config['eval_num'])


    print(f"Counts: (Total of {my_config['eval_num']} rollouts)")
