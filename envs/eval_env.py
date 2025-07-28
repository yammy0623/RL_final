import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch

# from diffusers_old import DDIMPipeline, DDIMScheduler, UNet2DModel
import os
from PIL import Image
from skimage.metrics import structural_similarity
from gymnasium.spaces import Box, Dict
import os
import random

# DDRM
from ddrm.datasets import get_dataset, data_transform, inverse_data_transform
from ddrm.functions.denoising import (
    initialize_generalized_steps,
    denoise_single_step,
    denoise_guided_addnoise,
)
import pdb
import copy
import gc
import csv


class EvalDiffusionEnv(gym.Env):
    def __init__(
        self,
        runner,
        gpu_idx,
        target_steps=10,
        max_steps=100,
        agent1=None,
        discrete_space=100,
    ):
        super(EvalDiffusionEnv, self).__init__()

        self.img_idx_so_far = 0

        # Model
        self.gpu_idx = gpu_idx
        self.device = torch.device("cuda:" + str(gpu_idx))
        self.last_T = 999
        self.runner = copy.deepcopy(runner)
        model, cls = self.runner.get_model()
        self.target_steps = target_steps
        self.final_threshold = 0.9
        _, _, sigma_0, config, deg, H_funcs, cls_fn = self.runner.sample(cls)
        # self.val_loader = val_loader
        self.sigma_0 = sigma_0
        self.config = config
        self.deg = deg
        self.H_funcs = H_funcs
        self.model = model.to(self.device)
        self.cls_fn = cls_fn
        self.sample_size = config.data.image_size
        self.batch_size = config.sampling.batch_size
        self.max_steps = max_steps
        self.valdata_len = self.runner.val_datalen

        # RL Setting
        self.agent1 = agent1  # RL model from subtask 1
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000 // self.target_steps)][::-1]
        self.adjust = True if agent1 is not None else False

        # skip = self.runner.num_timesteps // self.runner.args.timesteps
        # self.interval = self.runner.num_timesteps // target_steps
        # seq = range(0, self.runner.num_timesteps, skip)
        # seq_next = [-1] + list(seq[:-1])

        # Count the number of steps
        self.discrete_space = discrete_space
        if agent1 is None:  # Subtask 1
            if self.discrete_space == 0:
                self.action_space = gym.spaces.Box(
                    low=0, high=1
                )  # Continuous action space
            else:
                self.action_space = spaces.Discrete(
                    discrete_space
                )  # Discrete action space
            self.observation_space = Dict(
                {
                    "image": Box(
                        low=-1,
                        high=1,
                        shape=(3, self.sample_size, self.sample_size),
                        dtype=np.float32,
                    ),
                    "value": Box(
                        low=np.array([0]), high=np.array([999]), dtype=np.uint16
                    ),
                }
            )
        else:  # Subtask 2
            self.action_space = gym.spaces.Box(low=-5, high=5)
            self.observation_space = Dict(
                {
                    "image": Box(
                        low=-1,
                        high=1,
                        shape=(3, self.sample_size, self.sample_size),
                        dtype=np.float32,
                    ),
                    "value": Box(
                        low=np.array([0]), high=np.array([999]), dtype=np.uint16
                    ),
                    "remain": Box(
                        low=np.array([0]), high=np.array([999]), dtype=np.uint16
                    ),
                }
            )

        # Define the action and observation space
        self.observation_space = Dict(
            {
                "image": Box(
                    low=-1,
                    high=1,
                    shape=(3, self.sample_size, self.sample_size),
                    dtype=np.float32,
                ),
                "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16),
            }
        )
        self.data_idx = 0
        del runner
        with torch.cuda.device(self.gpu_idx):
            torch.cuda.empty_cache()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Reset counter, sequence
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []

        self.GT_image, self.classes = self.runner.test_dataset[self.img_idx_so_far]
        if self.GT_image.dim() == 3:
            self.GT_image = self.GT_image.unsqueeze(0)

        # noise and low level image y_0,
        self.noise_image, self.y_0, self.pinv_y_0, self.H_inv_y, self.GT_image = (
            self.runner.sample_init(
                self.GT_image,
                self.sigma_0,
                self.config,
                self.deg,
                self.H_funcs,
                self.model,
            )
        )

        # Initialization, extract degradation information from y_0 sigma 0, and H_func
        self.state = initialize_generalized_steps(
            self.device,
            self.pinv_y_0.to(self.device),
            self.last_T,
            self.runner.betas,
            self.H_funcs,
            self.y_0,
            self.sigma_0,
        )
        # self.x0_t = self.state['x']

        self.x0_t = self.pinv_y_0.to(self.device)

        observation = {"image": self.x0_t[0].cpu(), "value": np.array([self.last_T])}

        if self.agent1 is not None:

            with torch.no_grad():
                action, _state = self.agent1.predict(observation, deterministic=True)
                start_t = 50 * (1 + action) - 1
                next_t = torch.tensor(int(max(0, min(start_t, 999))))
                self.interval = int(next_t / (self.target_steps - 1))
                self.state["x"] = denoise_guided_addnoise(
                    self.state,
                    next_t,
                    self.et,
                    self.x0_t,
                    self.H_funcs,
                    self.sigma_0,
                    self.runner.args,
                )
                self.action_sequence.append(action.item())

                # Next round
                self.t = next_t
                self.x0_t, self.at, self.et = denoise_single_step(
                    self.state, self.model, self.t, self.cls_fn, self.classes
                )
                self.time_step_sequence.append(self.t.item())
                self.current_step_num += 1
                observation = {
                    "image": self.x0_t.cpu(),
                    "value": np.array([self.t]),
                    "remain": np.array([self.target_steps]),
                }

        with torch.cuda.device(self.gpu_idx):
            torch.cuda.empty_cache()  # Clear GPU cache
        return observation, {}

    def step(self, action):
        truncate = True if self.current_step_num >= self.max_steps else False

        with torch.no_grad():
            if self.agent1 is None:
                action = torch.tensor(action)

                if self.discrete_space == 0:
                    start_t = action * 999  # continuous action space
                else:
                    start_t = 1000 // self.discrete_space * (1 + action) - 1
                next_t = torch.tensor(int(max(0, min(start_t, 999))))
                self.interval = int(next_t / (self.target_steps - 1))
                self.state["x"] = denoise_guided_addnoise(
                    self.state,
                    self.t,
                    self.et,
                    self.x0_t,
                    self.H_funcs,
                    self.sigma_0,
                    self.runner.args,
                )

                # Next round
                self.x0_t, self.et = denoise_single_step(
                    self.state, self.model, self.t, self.cls_fn, self.classes
                )
                self.time_step_sequence.append(self.t.item())
                self.action_sequence.append(action.item())

                # denoising till the end for first step
                for i in range(self.target_steps - 1):
                    t = start_t - self.interval * (i + 1)
                    t = torch.tensor(int(max(0, min(t, 999))))
                    self.time_step_sequence.append(t.item())
                    self.state["x"] = denoise_guided_addnoise(
                        self.state,
                        self.t,
                        self.et,
                        self.x0_t,
                        self.H_funcs,
                        self.sigma_0,
                        self.runner.args,
                    )
                    self.x0_t, self.et = denoise_single_step(
                        self.state, self.model, self.t, self.cls_fn, self.classes
                    )
                    self.current_step_num += 1

            else:
                initial_t = (
                    self.t - self.interval if self.current_step_num != 0 else self.t
                )
                next_t = initial_t - self.interval * action
                thres = (
                    999 if self.current_step_num == 0 else self.time_step_sequence[-1]
                )
                next_t = torch.tensor(int(max(0, min(next_t, thres))))
                self.interval = (
                    int(next_t / (self.target_steps - self.current_step_num - 1))
                    if (self.target_steps - self.current_step_num - 1) != 0
                    else self.interval
                )

                self.state["x"] = denoise_guided_addnoise(
                    self.state,
                    next_t,
                    self.et,
                    self.x0_t,
                    self.H_funcs,
                    self.sigma_0,
                    self.runner.args,
                )

                self.t = next_t
                self.x0_t, self.et = denoise_single_step(
                    self.state, self.model, self.t, self.cls_fn, self.classes
                )

                self.action_sequence.append(action.item())
                self.time_step_sequence.append(self.t.item())

        # Finish the episode if denoising is done
        done = self.current_step_num == self.target_steps - 1
        # Calculate reward
        reward, ssim, psnr = self.calculate_reward(done)

        if done:
            self.runner.save_img(self.x0_t, self.img_idx_so_far)
            self.img_idx_so_far += (
                1 if self.img_idx_so_far < len(self.runner.test_dataset) - 1 else 0
            )

            # write inference result
            output_file = os.path.join(self.runner.args.image_folder, "results.csv")
            file_exists = os.path.isfile(output_file)
            with open(output_file, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(
                        ["Image Index", "Time Step Sequence", "SSIM", "PSNR"]
                    )
                writer.writerow(
                    [
                        self.img_idx_so_far,
                        str(self.time_step_sequence),
                        f"{ssim:.3f}",
                        f"{psnr:.3f}",
                    ]
                )

        info = {
            "ddim_t": self.uniform_steps[self.current_step_num],
            "t": self.t,
            "reward": reward,
            "ssim": ssim,
            "psnr": psnr,
            "time_step_sequence": self.time_step_sequence,
            "action_sequence": self.action_sequence,
            "threshold": self.final_threshold,
        }

        if self.agent1 is None:
            observation = {"image": self.x0_t[0].cpu(), "value": np.array([self.t])}
        else:
            observation = {
                "image": self.x0_t[0].cpu(),
                "value": np.array([self.t]),
                "remain": np.array([self.target_steps - self.current_step_num - 1]),
            }

        self.current_step_num += 1

        return observation, reward, done, truncate, info

    def calculate_reward(self, done):
        reward = 0
        x = inverse_data_transform(self.config, self.x0_t).to(self.runner.device)
        # orig = self.GT_image.to(self.runner.device)
        orig = inverse_data_transform(self.config, self.GT_image).to(self.runner.device)
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        # ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        ssim = structural_similarity(
            x.squeeze(0).cpu().numpy(),
            orig.squeeze(0).cpu().numpy(),
            win_size=21,
            channel_axis=0,
            data_range=1.0,
        )
        # Sparse reward (SSIM)
        if done and ssim > self.final_threshold:
            reward += 1

        return reward, ssim, psnr
