from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    # VecVideoRecorder,
    # SubprocVecEnv,
)
from stable_baselines3 import SAC, A2C
from gymnasium import spaces
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F
from func import MD_SAC, PPO, A2C
import math

import os

from ddrm.runners.diffusion import Diffusion
from arguments import parse_args_and_config
from torch.cuda.amp import autocast



LOG = False
warnings.filterwarnings("ignore")
register(
    id="final-v0",
    entry_point="envs:DiffusionEnv",
    # kwargs={'model_name': 'default_model_name', 'target_steps': 10, 'max_steps': 100}
)


def make_env(my_config):
    def _init():
        config = {
            "runner": my_config["runner"],
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
        }
        return gym.make("final-v0", **config)

    return _init


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['image'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space['image'].sample()[None]).float()
            ).shape[1]

        self.fc = nn.Linear(1, 32)
        self.linear = nn.Sequential(nn.Linear(n_flatten + 32, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        img_features = self.cnn(observations['image'].float())
        value_features = F.relu(self.fc(observations['value'].float()))
        combined = th.cat([img_features, value_features], dim=1)
        return self.linear(combined)

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_reward = 0
    avg_ssim = 0
    avg_ddim_ssim = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs, _ = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        avg_reward += info['reward']
        avg_ssim   += info['ssim']
        avg_ddim_ssim += info['ddim_ssim']

    avg_reward /= eval_episode_num
    avg_ssim /= eval_episode_num
    avg_ddim_ssim /= eval_episode_num
        
    return avg_reward, avg_ssim, avg_ddim_ssim, info['time_step_sequence']

def train(eval_env, rl_model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_ssim = 0


    for epoch in range(config["epoch_num"]):

        # Uncomment to enable wandb logging
        if LOG:
            rl_model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                callback=WandbCallback(
                    gradient_save_freq=100,
                    verbose=2,
                ),
            )
        else:
            with autocast():
                rl_model.learn(
                    total_timesteps=config["timesteps_per_epoch"],
                    reset_num_timesteps=False,
                    progress_bar=True,
                )

        th.cuda.empty_cache()  # Clear GPU cache
        
        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_reward, avg_ssim, avg_ddim_ssim, time_step_sequence = eval(eval_env, rl_model, config["eval_episode_num"])

        print("Avg_reward:  ", avg_reward)
        print("Avg_ssim:    ", avg_ssim)
        print("Avg_ddim_ssim:", avg_ddim_ssim)
        print("Time_step_sequence:", time_step_sequence)
        print()
        print("---------------")

        ### Save best model
        if current_best_ssim < avg_ssim:
            print("Saving Model !!!")
            current_best_ssim = avg_ssim
            save_path = config["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            rl_model.save(f"{save_path}/best")


        if LOG:
            wandb.log(
                {
                    "avg_reward": avg_reward,
                    "avg_ssim": avg_ssim,
                    "ddim_ssim": ddim_ssim,
                }
            )
        

def main():


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )

    # Create diffusion model
    # TODO: change to read yaml
    args, config = parse_args_and_config()
    runner = Diffusion(args, config)

    # my_config = {
    #     "algorithm": MD_SAC,
    #     "buffer_size": 10000, # for SAC only, default is 1e6.
    #     "num_train_envs": 16,
    #     "policy_network": "MultiInputPolicy",
    #     "epoch_num": 500,
    #     "timesteps_per_epoch": 100,
    #     "eval_episode_num": 16,
    #     "learning_rate": 1e-4,
    #     "policy_kwargs": policy_kwargs,
    #     "runner": runner,
    #     "target_steps": args.target_steps,
    #     "max_steps": 100,
    # }

    my_config = {
        "run_id": "SAC_v1",

        "algorithm": A2C,
        "policy_network": "MultiInputPolicy",
        "save_path": "model/sample_model",

        "epoch_num": 500,
        "timesteps_per_epoch": 100,
        "eval_episode_num": 10,
        "learning_rate": 1e-4,
        "policy_kwargs": policy_kwargs,

        "DM_model": "model/ddpm_ema_cifar10",
        "target_steps": 10,
        "max_steps": 100,

        "num_train_envs": 8,
        "runner": runner,
    }

    my_config['run_id'] = f'SR_baseline_A2C_env_{my_config["num_train_envs"]}_steps_{my_config["target_steps"]}'
    my_config['save_path'] = f'model/SR_baseline_A2C_{my_config["target_steps"]}'

    if LOG:
        _ = wandb.init(
            project="final",
            config=my_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            id=my_config["run_id"],
        )

    config = {
            "runner": my_config["runner"],
            "target_steps": my_config["target_steps"],
            "max_steps": my_config["max_steps"],
        }
    # Create training environment
    num_train_envs = my_config["num_train_envs"]
    train_env = DummyVecEnv([make_env(config) for _ in range(num_train_envs)])

    # Create evaluation environment
    # eval_env = DummyVecEnv([make_env(my_config)])
    # TODO: Why using SB3 API?
    # Create evaluation environment (via SB3 API) 
    eval_env = gym.make('final-v0', **config)

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    rl_model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=2,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"],
        # buffer_size=my_config["buffer_size"]
    )

    train(eval_env, rl_model, my_config)

if __name__ == "__main__":
    main()
