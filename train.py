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
from torch.cuda.amp import autocast, GradScaler
# from new_A2C_model import MixedPrecisionA2C
scaler = GradScaler()

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
            "agent1": my_config["agent1"],
        }
        return gym.make("final-v0", **config)

    return _init


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, use_scale_shift_norm: bool = True):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space["image"].shape[0]
        n_input_channels = 3
        self.use_scale_shift_norm = use_scale_shift_norm
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
        self.embedding_output = nn.Linear(32, features_dim * 2)
        self.out_norm = nn.Linear(n_flatten, features_dim)  # Normalizing layer
        self.out_rest = nn.Sequential(
            nn.Linear(features_dim, features_dim),  # Further processing layer
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        img_features = self.cnn(observations['image'].float())
        value_features = F.relu(self.fc(observations['value'].float()))
        if self.use_scale_shift_norm:
            emb_out = self.embedding_output(value_features)
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.out_norm(img_features) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = self.out_rest(self.out_norm(img_features + value_features))
        return h

def eval(env, rl_model, eval_episode_num, args):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_reward = 0
    avg_reward_t = [0 for _ in range(args.target_steps)]
    avg_ssim = 0
    avg_psnr = 0
    ddim_ssim = 0
    ddim_psnr = 0
    avg_start_t = 0

    with th.no_grad():
        for seed in range(eval_episode_num):
            done = False
            # Set seed using old Gym API
            # env.seed(seed)
            # obs = env.reset()
            obs, info = env.reset(seed=seed)
            now_t = 0
            # Interact with env using old Gym API
            while not done:
                action, _state = rl_model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                avg_reward_t[now_t] += info['reward']
                now_t += 1

            avg_reward += info['reward']
            avg_ssim   += info['ssim']
            avg_psnr += info['psnr']
            ddim_ssim += info['ddim_ssim']
            ddim_psnr += info['ddim_psnr']
            avg_start_t += info['time_step_sequence'][0]


    avg_reward /= eval_episode_num
    avg_ssim /= eval_episode_num
    avg_psnr /= eval_episode_num
    ddim_ssim /= eval_episode_num
    ddim_psnr /= eval_episode_num
    avg_start_t /= eval_episode_num
    for i in range(5):
        avg_reward_t[i] = avg_reward_t[i] / eval_episode_num
    
    return avg_reward, avg_ssim, avg_psnr, ddim_ssim, ddim_psnr, info['time_step_sequence'], info['action_sequence'], avg_reward_t, avg_start_t

def train(eval_env, rl_model, config, epoch_num, args, second_stage=False):
    """Train agent using SB3 algorithm and my_config"""
    current_best_psnr = 0
    current_best_ssim = 0


    for epoch in range(epoch_num):

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
            rl_model.learn(
                total_timesteps=config["timesteps_per_epoch"],
                reset_num_timesteps=False,
                progress_bar=True,
            )

        th.cuda.empty_cache()  # Clear GPU cache
        
        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_reward, avg_ssim, avg_psnr, ddim_ssim, ddim_psnr, time_step_sequence, action_sequence, avg_reward_t, avg_start_t = eval(
            eval_env, rl_model, config["eval_episode_num"], args
        )
        print("---------------")

        ### Save best model
        if current_best_psnr < avg_psnr and current_best_ssim < avg_ssim:# and epoch > 10:
            print("Saving Model !!!")
            current_best_psnr = avg_psnr
            current_best_ssim = avg_ssim
            save_path = config["save_path"]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if second_stage:
                rl_model.save(f"{save_path}/best_2")
            else:
                rl_model.save(f"{save_path}/best")


        print("Avg_reward:  ", avg_reward)
        print("Avg_reward_t:  ", avg_reward_t)
        print("Avg_start_t:  ", avg_start_t)
        print("Avg_ssim:    ", avg_ssim)
        print("Avg_psnr:    ", avg_psnr)
        print("Current_best_ssim:", current_best_ssim)
        print("Current_best_psnr:", current_best_psnr)
        print("DDIM_ssim:   ", ddim_ssim)
        print("DDIM_psnr:   ", ddim_psnr)
        print("Time_step_sequence:", time_step_sequence)
        print("Action_sequence:", action_sequence)
        print()


        if LOG:
            wandb.log(
                {
                    "avg_reward": avg_reward,
                    "avg_ssim": avg_ssim,
                    "avg_psnr": avg_psnr,
                    "ddim_ssim": ddim_ssim,
                    "ddim_psnr": ddim_psnr,
                    "start_t": avg_start_t,
                }
            )
        


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


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Create diffusion model
    # TODO: change to read yaml
    args, config = parse_args_and_config()
    runner = Diffusion(args, config)

    
    # DDPG start
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    # handler1.setFormatter(formatter)
    # logger = logging.getLogger()
    # logger.addHandler(handler1)
    # logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
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

    log_path = os.path.join(args.image_folder, '0_logs.log')
    fh = logging.FileHandler(log_path)#, mode='a')
    fh.setFormatter(formatter)
    # logger.setLevel(level)
    # logger.addHandler(fh)
    # for arg, value in sorted(vars(args).items()):
    #     logger.info("Argument %s: %r", arg, value)

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


    # END

    my_config = {
        "run_id": "SAC_v1",

        "algorithm": A2C,
        "policy_network": "MultiInputPolicy",
        "save_path": "model/sample_model",
        "first_stage_epoch_num": 50,

        "epoch_num": 500,
        "timesteps_per_epoch": 100,
        "eval_episode_num": 10,
        "learning_rate": 1e-4,
        "policy_kwargs": policy_kwargs,

        # "DM_model": "model/ddpm_ema_cifar10",
        "target_steps": args.target_steps-1,
        "max_steps": 100,

        "num_train_envs": 8,
        "runner": runner,
    }

    my_config['run_id'] = f'new_SR_2agent_A2C_env_{my_config["num_train_envs"]}_steps_{args.target_steps}'
    my_config['save_path'] = f'model/new_SR_2agent_A2C_{args.target_steps}'

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
            "agent1": None,
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
        # device="cpu"
        # buffer_size=my_config["buffer_size"]
    )

    if args.second_stage == False:
        ### First stage training
        train(eval_env, rl_model, my_config, epoch_num = my_config["first_stage_epoch_num"], args=args)
    else:
        ### Second stage training
        print("Loaded model from: ", f"{my_config['save_path']}/best")
        rl_model = my_config["algorithm"].load(f"{my_config['save_path']}/best")
        config['agent1'] = rl_model

        train_env = DummyVecEnv([make_env(config) for _ in range(num_train_envs)])
        eval_env = gym.make('final-v0', **config)
        rl_model_2 = my_config["algorithm"](
            my_config["policy_network"], 
            train_env, 
            verbose=2,
            tensorboard_log=my_config["run_id"],
            learning_rate=my_config["learning_rate"],
            policy_kwargs=my_config["policy_kwargs"],
        )
        train(eval_env, rl_model_2, my_config, epoch_num = my_config["epoch_num"] - my_config["first_stage_epoch_num"], args=args, second_stage=True)

if __name__ == "__main__":
    main()
