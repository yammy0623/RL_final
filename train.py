from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    # VecVideoRecorder,
    # SubprocVecEnv,
)
from stable_baselines3 import SAC
from gymnasium import spaces
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
import torch.nn.functional as F
from func import MD_SAC
import torch
import argparse
import logging
import yaml
import os
import numpy as np
import shutil
import sys
import math
from ddrm.runners.diffusion import Diffusion


LOG = False
warnings.filterwarnings("ignore")
register(
    id="final-v0",
    entry_point="envs:DiffusionEnv",
    # kwargs={'model_name': 'default_model_name', 'target_steps': 10, 'max_steps': 100}
)


# def make_env(my_config):
#     env = gym.make('final-v0')#, model_name=my_config["DM_model"], target_steps=my_config["target_steps"], max_steps=my_config["max_steps"])
#     return env


def make_env(my_config):
    def _init():
        config = {
            "runner": my_config["runner"],
            "model": my_config["diff_model"],
            "cls": my_config["diff_cls"],
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
        # batch_size = observation_space["lower_dim_image"].shape[0]
        # res = math.sqrt(observation_space["lower_dim_image"].shape[1]//3)
        # self.obs_restored = observation_space["lower_dim_image"].reshape(
        #     batch_size, 3, res, res
        # )
        n_input_channels = observation_space["image"].shape[0]
        n_input_channels = 3
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
                th.as_tensor(observation_space["image"].sample()[None]).float()
            ).shape[1]

        self.fc = nn.Linear(1, 32)
        self.linear = nn.Sequential(nn.Linear(n_flatten + 32, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        img_features = self.cnn(observations["image"].float())
        value_features = F.relu(self.fc(observations["value"].float()))
        combined = th.cat([img_features, value_features], dim=1)
        return self.linear(combined)


def eval(env, rl_model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_reward = 0
    avg_ssim = 0
    avg_ddim_ssim = 0
    for seed in range(eval_episode_num):
        done = False
        # Set seed using old Gym API
        env.seed(seed)
        obs = env.reset()

        # Interact with env using old Gym API
        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)

        avg_reward += info[0]["reward"]
        avg_ssim += info[0]["ssim"]
        avg_ddim_ssim += info[0]["ddim_ssim"]

    avg_reward /= eval_episode_num
    avg_ssim /= eval_episode_num
    avg_ddim_ssim /= eval_episode_num

    return avg_reward, avg_ssim, avg_ddim_ssim, info[0]["time_step_sequence"]


def train(eval_env, rl_model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best = 0
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

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_reward, avg_ssim, avg_ddim_ssim, time_step_sequence = eval(
            eval_env, rl_model, config["eval_episode_num"]
        )

        print("Avg_reward:  ", avg_reward)
        print("Avg_ssim:    ", avg_ssim)
        print("Avg_ddim_ssim:", avg_ddim_ssim)
        print("Time_step_sequence:", time_step_sequence)
        print()
        if LOG:
            wandb.log(
                {
                    "avg_reward": avg_reward,
                    "avg_ssim": avg_ssim,
                    "avg_ddim_ssim": avg_ddim_ssim,
                }
            )

        ### Save best model
        if current_best < avg_ssim:
            print("Saving Model")
            current_best = avg_ssim
            save_path = config["save_path"]
            rl_model.save(f"{save_path}/{epoch}")

        print("---------------")


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="ddrm/exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument("--deg", type=str, required=True, help="Degradation")
    parser.add_argument("--sigma_0", type=float, required=True, help="Sigma_0")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta")
    parser.add_argument("--etaB", type=float, default=1, help="Eta_b (before)")
    parser.add_argument("--subset_start", type=int, default=-1)
    parser.add_argument("--subset_end", type=int, default=-1)

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("ddrm/configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(args.exp, "image_samples", args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    # TODO: debug
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=32),
    )

    # Create diffusion model
    # TODO: change to read yaml
    args, config = parse_args_and_config()
    runner = Diffusion(args, config)
    diff_model, cls = runner.get_model()
    print("cls = ", cls)
    my_config = {
        "run_id": "SAC_v1",
        "algorithm": MD_SAC,
        "policy_network": "MultiInputPolicy",
        "save_path": "model/sample_model",
        "epoch_num": 500,
        "timesteps_per_epoch": 100,
        "eval_episode_num": 10,
        "learning_rate": 1e-4,
        "policy_kwargs": policy_kwargs,
        "runner": runner,
        "diff_model": diff_model,
        "diff_cls": cls,
        # "DM_model": "model/ddpm_ema_cifar10",
        "target_steps": 10,
        "max_steps": 100,
        "num_train_envs": 1,
    }
    if LOG:
        _ = wandb.init(
            project="final",
            config=my_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            id=my_config["run_id"],
        )

    # Create training environment
    num_train_envs = my_config["num_train_envs"]
    train_env = DummyVecEnv([make_env(my_config) for _ in range(num_train_envs)])
    # train_env = SubprocVecEnv([make_env(my_config) for _ in range(num_train_envs)])

    # env = DiffusionEnv('google/ddpm-cifar10-32')
    # model = SAC("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    # model.learn(total_timesteps=20000)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(my_config)])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    rl_model = my_config["algorithm"](
        my_config["policy_network"],
        train_env,
        verbose=2,
        tensorboard_log=my_config["run_id"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"],
    )

    train(eval_env, rl_model, my_config)

    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     print('Train info:', info)
    #     env.render()
    #     if done:
    #         obs = env.reset()


if __name__ == "__main__":
    main()
