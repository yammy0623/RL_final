import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import torch
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


# from save_img import save_image
class DiffusionEnv(gym.Env):
    def __init__(
        self, runner, gpu_idx, env_id, args, target_steps=10, max_steps=100, agent1=None
    ):
        super(DiffusionEnv, self).__init__()
        self.isShowfig = False
        self.runner = copy.deepcopy(runner)
        model, cls = self.runner.get_model()
        self.env_id = env_id

        # Model
        self.gpu_idx = gpu_idx
        self.device = torch.device("cuda")
        self.last_T = 999
        _, _, sigma_0, config, deg, H_funcs, cls_fn = self.runner.sample(cls)
        self.sigma_0 = sigma_0
        self.config = config
        self.args = args
        self.deg = deg
        self.H_funcs = H_funcs
        self.model = model.to(self.device)
        self.cls_fn = cls_fn
        self.sample_size = config.data.image_size
        self.max_steps = max_steps

        # RL Setting
        self.agent1 = agent1  # RL model from subtask 1
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000 // self.target_steps)][::-1]
        self.adjust = True if agent1 is not None else False

        # ddim sequence
        skip = self.runner.num_timesteps // self.runner.args.timesteps
        self.interval = self.runner.num_timesteps // target_steps
        seq = range(0, self.runner.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        self.ddim_seq = list(reversed(seq))
        self.ddim_seq_next = list(reversed(seq_next))

        # sequence
        self.time_step_sequence = []
        self.action_sequence = []

        # Count the number of steps
        self.current_step_num = 0

        # Define the action and observation space
        if self.adjust:  # Subtask 2
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
                }
            )
        else:  # Subtask 1
            self.action_space = spaces.Discrete(20)  # Discrete action space
            # self.action_space = gym.spaces.Box(low=-1, high=1) # Continuous action space
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

        # Initialize the random seed
        self.seed(232)
        del runner
        with torch.cuda.device(self.gpu_idx):
            torch.cuda.empty_cache()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

    def reset(self, seed=None, options=None):
        # print("env id: ", self.env_id)
        if seed is not None:
            self.seed(seed)

        # Reset counter, sequence
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []

        self.data_idx = random.randint(0, len(self.runner.train_dataset) - 1)
        # print("data idx:", self.data_idx)
        self.GT_image, self.classes = self.runner.train_dataset[self.data_idx]
        if self.GT_image.dim() == 3:
            self.GT_image = self.GT_image.unsqueeze(0)

        # Load Image
        # self.data_iter = iter(self.train_loader)
        # self.GT_image, self.classes = next(self.data_iter)

        # noise and low level image y_0,
        self.noise_image, self.y_0, self.pinv_y_0, self.H_inv_y, self.GT_image = (
            self.runner.sample_init(
                self.GT_image,
                self.sigma_0,
                self.config,
                self.deg,
                self.H_funcs,
            )
        )

        # save_image(inverse_data_transform(self.config, self.GT_image).to(self.runner.device), output_path="./results", file_name="self.GT_image.png")
        # save_image(inverse_data_transform(self.config, self.pinv_y_0).to(self.runner.device), output_path="./results", file_name="self.pinv_y_0.png")
        if self.isShowfig:
            save_img(self.GT_image, self.config, "GT_img")
            save_img(self.pinv_y_0, self.config, "y0")
        # save_img(self.pinv_y_0, self.config, "self.pinv_y_0")
        # save_img(self.y_0, self.config, "y_0")
        # Initialization, extract degradation information from y_0 sigma 0, and H_func
        """
        state = {
                    "x": x, (mapping to y_0 space)
                    "b": b (beta)
                    "Sigma": Sigma,
                    "Sig_inv_U_t_y": Sig_inv_U_t_y,
                    "U_t_y": U_t_y,
                    "singulars": singulars,
                    "large_singulars_index": large_singulars_index,
                }
        """

        if self.args.start_with == "y_addnoise":
            first_x = self.pinv_y_0.to(self.device).clone()
        else:
            first_x = self.noise_image.clone()

        self.state = initialize_generalized_steps(
            self.device,
            first_x,
            self.last_T,
            self.runner.betas,
            self.H_funcs,
            self.y_0,
            self.sigma_0,
        )

        # print(self.ddim_seq[0])
        self.ddim_state = initialize_generalized_steps(
            self.device,
            self.noise_image,
            self.ddim_seq[0],
            self.runner.betas,
            self.H_funcs,
            self.y_0,
            self.sigma_0,
        )

        # DDIM should start from noise!!
        with torch.no_grad():
            for i in range(self.target_steps):
                ddim_t = torch.tensor(self.uniform_steps[i])
                # print(f"ddim_t = {ddim_t}")
                if i != 0:
                    self.ddim_state["x"] = denoise_guided_addnoise(
                        self.ddim_state,
                        ddim_t,
                        ddim_et,
                        ddim_x0_t,
                        self.H_funcs,
                        self.sigma_0,
                        self.runner.args,
                    )
                ddim_x0_t, ddim_et = denoise_single_step(
                    self.ddim_state, self.model, ddim_t, self.cls_fn, self.classes
                )

                if self.isShowfig:
                    save_img(ddim_x0_t, self.config, "ddim_x0_t")
                # save_img(self.ddim_state['x'], self.config, "ddim_xt_next")

        orig = inverse_data_transform(self.config, self.GT_image[0]).to(
            self.runner.device
        )
        ddim_x = inverse_data_transform(self.config, ddim_x0_t[0]).to(
            self.runner.device
        )
        ddim_mse = torch.mean((ddim_x - orig) ** 2)
        self.ddim_psnr = 10 * torch.log10(1 / ddim_mse).item()
        self.ddim_ssim = structural_similarity(
            ddim_x.squeeze(0).cpu().numpy(),
            orig.squeeze(0).cpu().numpy(),
            win_size=21,
            channel_axis=0,
            data_range=1.0,
        )
        print("ddim PSNR: %.2f, SSIM: %.2f" % (self.ddim_psnr, self.ddim_ssim))

        _, self.et = denoise_single_step(
            self.state, self.model, self.last_T, self.cls_fn, self.classes
        )
        self.x0_t = first_x
        self.t = self.last_T
        observation = {"image": self.x0_t[0].cpu(), "value": np.array([self.t])}
        # TODO START HERE
        if self.adjust:  # Second subtask
            action, _state = self.agent1.predict(observation, deterministic=True)
            self.action_sequence.append(action.item())

            start_t = 50 * (1 + action) - 1
            next_t = torch.tensor(int(max(0, min(start_t, 999))))
            self.t = next_t
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
            self.current_step_num += 1

            observation = {"image": self.x0_t[0].cpu(), "value": np.array([self.t])}

        with torch.cuda.device(self.gpu_idx):
            torch.cuda.empty_cache()  # Clear GPU cache

        return observation, {}

    def step(self, action):
        truncate = self.current_step_num >= self.max_steps
        with torch.cuda.device(self.gpu_idx):
            torch.cuda.empty_cache()
        # RL
        with torch.no_grad():
            if self.adjust == False:
                start_t = 50 * (1 + action) - 1
                next_t = torch.tensor(int(max(0, min(start_t, 999))))
                self.interval = int(next_t / (self.target_steps - 1))
                # need to add noise from y

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
            else:  # Second subtask
                next_t = self.t - self.interval - self.interval * action
                next_t = torch.tensor(int(max(0, min(next_t, 999))))
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
                self.action_sequence.append(action.item())

        self.t = next_t
        self.x0_t, self.et = denoise_single_step(
            self.state, self.model, self.t, self.cls_fn, self.classes
        )

        self.time_step_sequence.append(self.t.item())
        print(self.time_step_sequence)

        # Run remaining steps via uniform policy
        self.uniform_x0_t = self.x0_t.clone()
        uniform_et = self.et.clone()
        # self.uniform_state = self.state.clone()
        self.uniform_state = copy.deepcopy(self.state)

        for i in range(self.target_steps - self.current_step_num - 1):
            uniform_t = torch.tensor(int(self.t - self.interval - self.interval * i))
            uniform_t = torch.tensor(max(0, min(uniform_t, 999)))
            if (
                self.args.start_with == "y_addnoise"
                or self.args.start_with == "rand_addnoise"
                or not i == 0
            ):
                self.uniform_state["x"] = denoise_guided_addnoise(
                    self.uniform_state,
                    uniform_t,
                    uniform_et,
                    self.uniform_x0_t,
                    self.H_funcs,
                    self.sigma_0,
                    self.runner.args,
                )
            self.uniform_x0_t, uniform_et = denoise_single_step(
                self.uniform_state, self.model, uniform_t, self.cls_fn, self.classes
            )
            if self.isShowfig:
                save_img(self.uniform_x0_t, self.config, "uniform_x0_t")
            # save_img(self.uniform_state['x'], self.config, "uniform_xt_next")

        # Finish the episode if denoising is done
        done = (self.current_step_num == self.target_steps - 1) or not self.adjust
        reward, ssim, psnr, ddim_ssim, ddim_psnr = self.calculate_reward(done)

        info = {
            "ddim_t": self.uniform_steps[self.current_step_num],
            "t": self.t,
            "reward": reward,
            "ssim": ssim,
            "psnr": psnr,
            "ddim_ssim": ddim_ssim,
            "ddim_psnr": ddim_psnr,
            "time_step_sequence": self.time_step_sequence,
            "action_sequence": self.action_sequence,
        }
        # print('info:', info)
        if self.adjust:
            observation = {
                "image": self.x0_t[0].cpu(),
                # "image2": self.Apy[0].cpu(),
                "value": np.array([self.t]),
            }
        else:
            observation = {"image": self.x0_t[0].cpu(), "value": np.array([self.t])}
        self.current_step_num += 1

        with torch.cuda.device(self.gpu_idx):
            torch.cuda.empty_cache()
        return observation, reward, done, truncate, info

    def calculate_reward(self, done):
        reward = 0
        if self.isShowfig:
            save_img(self.uniform_x0_t[0], self.config, "uniform_x0_t")
            save_img(self.GT_image[0], self.config, "GT_img")
        orig = inverse_data_transform(self.config, self.GT_image[0]).to(
            self.runner.device
        )
        # orig = self.GT_image[0].to(self.runner.device)
        if done and self.adjust:
            x = inverse_data_transform(self.config, self.x0_t[0]).to(self.runner.device)
        else:
            x = inverse_data_transform(self.config, self.uniform_x0_t[0]).to(
                self.runner.device
            )

        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        ssim = structural_similarity(
            x.squeeze(0).cpu().numpy(),
            orig.squeeze(0).cpu().numpy(),
            win_size=21,
            channel_axis=0,
            data_range=1.0,
        )
        print("rl PSNR: %.2f, SSIM: %.2f" % (psnr, ssim))
        # Intermediate reward
        if not done:  # and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            reward += 0.5 / self.target_steps * (psnr / self.ddim_psnr)
            reward += 0.5 / self.target_steps * (ssim / self.ddim_ssim)

        # Sparse reward
        if done:  # and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            reward += 0.5 * (psnr / self.ddim_psnr)
            reward += 0.5 * (ssim / self.ddim_ssim)

        return reward, ssim, psnr, self.ddim_ssim, self.ddim_psnr

    def render(self, mode="human", close=False):
        # This could visualize the current state if necessary
        pass

    def set_adjust(self, adjust):
        self.adjust = adjust
        print(f"Set adjust to {adjust}")


import torchvision.utils as tvu


def save_img(x, config, name):
    x = inverse_data_transform(config, x)
    tvu.save_image(x, os.path.join("image", f"{name}.png"))
