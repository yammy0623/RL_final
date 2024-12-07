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
from ddrm.functions.denoising import initialize_generalized_steps, denoise_single_step
import pdb


class EvalDiffusionEnv(gym.Env):
    def __init__(
        self,
        runner,
        model,
        cls,
        target_steps=10,
        max_steps=100,
        img_save_path=None,
        action_range=1.0,
        seed=0,
    ):
        super(EvalDiffusionEnv, self).__init__()
        # Original Param
        self.sample_number_count = 0
        self.img_save_path = img_save_path
        self.target_steps = target_steps

        # DDRM
        self.runner = runner
        val_loader, sigma_0, config, deg, H_funcs, model, idx_so_far, cls_fn = (
            self.runner.sample(cls)
        )

        self.val_loader = val_loader
        self.sigma_0 = sigma_0
        self.config = config
        self.deg = deg
        self.H_funcs = H_funcs
        self.model = model
        self.idx_so_far = idx_so_far
        self.cls_fn = cls_fn
        self.model.to("cuda")

        self.valdata_len = self.runner.val_datalen
        self.current_image_idx = 0
        self.sample_size = config.data.image_size
        self.batch_size = config.sampling.batch_size

        self.data_iter = iter(self.val_loader)
        self.GT_image, self.classes = next(self.data_iter)

        self.model.to("cuda")
        self.sample_size = self.model.config.sample_size
        # RL steps
        # self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder=scheduler_subfolder)
        # self.scheduler.set_timesteps(max_steps)
        self.time_step_sequence = []
        # self.action_sequence = []
        self.max_steps = max_steps
        self.current_step_num = 0

        # DDIM steps
        # self.ddim_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder=scheduler_subfolder)
        # self.ddim_scheduler.set_timesteps(target_steps)
        skip = self.runner.num_timesteps // self.runner.args.timesteps
        self.interval = self.runner.num_timesteps // target_steps
        # seq = range(self.runner.num_timesteps, 0, -1*skip)
        seq = range(0, self.runner.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])
        self.ddim_seq = list(reversed(seq))
        self.ddim_seq_next = list(reversed(seq_next))

        # Maximum number of steps  (Baseline)
        self.max_steps = max_steps
        # Count the number of steps
        self.current_step_num = 0
        # Define the action and observation space
        self.action_space = gym.spaces.Box(
            low=-action_range, high=action_range, shape=(1,)
        )
        self.observation_space = Dict(
            {
                "image": Box(
                    low=0,
                    high=255,
                    shape=(3, self.sample_size, self.sample_size),
                    dtype=np.uint8,
                ),
                "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16),
            }
        )
        # Initialize the random seed
        self.seed(seed)
        self.episode_init = True
        self.state = None
        # Initialize with a random noisy image
        # self.current_image = torch.randn((1, 3, self.sample_size, self.sample_size), device="cuda", generator=self.generator)
        self.current_noise_image, y_0 = self.runner.sample_init(
            self.GT_image,
            self.sigma_0,
            self.config,
            self.deg,
            self.H_funcs,
            self.model,
            self.idx_so_far,
            self.cls_fn,
            self.classes,
        )
        self.y_0 = y_0
        self.ddim_current_image = self.current_noise_image.clone()

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        print("Set seed:", seed)
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        # return [seed]

    def reset(self, seed=None, options=None):
        self.episode_init = True
        if seed is not None:
            self.seed(seed)

        if self.current_image_idx < self.valdata_len - 1:
            # print("Current image index: ", self.current_image_idx)
            # print("Val data length: ", self.valdata_len)
            self.current_image_idx += 1

            self.current_step_num = 0
            self.time_step_sequence = []
            # self.action_sequence = []

            # Change to next image
            self._load_next_image()
        else:
            # print("All images are used. Resetting to the first image.")
            self.current_image_idx = 0
            self.data_iter._reset(self.val_loader)

        observation = {
            # "lower_dim_image": (self.current_noise_image).reshape(self.batch_size, 1).cpu().numpy(),
            "image": torch.tensor(self.current_noise_image).squeeze(0).cpu().numpy(),
            "value": np.array([999]),
        }
        return observation, {}

    def _load_next_image(self):
        self.GT_image, self.classes = next(self.data_iter)
        self.current_noise_image, self.y_0 = self.runner.sample_init(
            self.GT_image,
            self.sigma_0,
            self.config,
            self.deg,
            self.H_funcs,
            self.model,
            self.idx_so_far,
            self.cls_fn,
            self.classes,
        )

    def step(self, action):
        truncate = self.current_step_num >= self.max_steps

        # Denoise current image at time t
        interval = self.interval
        if self.episode_init:
            # Randomly pick last T in a certain range
            last_T = torch.randint(low=800, high=1000, size=(1,)).item()
            self.state = initialize_generalized_steps(
                self.current_noise_image.to("cuda"),
                last_T,
                self.runner.betas,
                self.H_funcs,
                self.y_0,
                self.sigma_0,
            )
            self.episode_init = False
            self.t = last_T
            # print("type of self.t: ", type(self.t))

        self._update_sequences(self.t, action)
        next_t = self._calculate_time_step(action, interval)
        self.current_noise_image = self._perform_denoising_single_step(
            self.state, self.t, next_t
        )

        # DDRM
        ddim_t = self.ddim_seq[self.current_step_num]
        # Finish the episode if denoising is done
        done = self.current_step_num == self.target_steps - 1
        # Increase number of steps
        self.current_step_num += 1
        # Calculate reward
        reward, ssim, ddim_ssim = self.calculate_reward(done)
        info = self._create_info_dict(ddim_t, self.t, reward, ssim, ddim_ssim)
        # print('info:', info)
        observation = {"image": self.current_noise_image.squeeze(0), "value": self.t}

        # Save the image if done
        if done and self.img_save_path is not None:
            # print(self.current_image)
            if not os.path.exists(self.img_save_path):
                os.makedirs(self.img_save_path)
            # print(info['time_step_sequence'])
            # print("timesteps", info['time_step_sequence'])
            images = (self.current_noise_image / 2 + 0.5).clamp(0, 1)
            images = images.cpu().permute(0, 2, 3, 1).numpy()[0]
            images = Image.fromarray((images * 255).round().astype("uint8"))
            filename = os.path.join(
                self.img_save_path, f"img_{self.sample_number_count}.png"
            )
            images.save(filename)
            # print(f"Image saved at {filename}")
            self.sample_number_count += 1
        return observation, reward, done, truncate, info

    def _calculate_time_step(self, action, interval):
        action = action.item()

        t = int(
            torch.round(
                torch.tensor(self.ddim_seq[self.current_step_num] - interval * action)
            )
        )
        # t = int(torch.round(torch.tensor(self.t - interval * action)))
        # print("t = ", t)
        final_t = max(0, min(t, 999))
        # print("final_t = ", final_t)
        return torch.tensor(final_t)

    def _update_sequences(self, t, action):
        self.time_step_sequence.append(t.item() if type(t) == torch.Tensor else t)
        self.action_sequence.append(action.item())

    def _perform_denoising_single_step(self, state, t, next_t):
        with torch.no_grad():
            xs, x0_preds = denoise_single_step(
                state,
                self.model,
                t,
                next_t,
                self.runner.betas,
                self.H_funcs,
                self.sigma_0,
                etaB=self.runner.args.etaB,
                etaA=self.runner.args.eta,
                etaC=self.runner.args.eta,
                cls_fn=self.cls_fn,
                classes=self.classes,
            )
            x = torch.stack([inverse_data_transform(self.config, y) for y in xs])
        return x

    def _create_info_dict(self, ddim_t, t, reward, ssim, ddim_ssim):
        return {
            "ddim_t": ddim_t,
            "t": t,
            "reward": reward,
            "ssim": ssim,
            "ddim_ssim": ddim_ssim,
            "time_step_sequence": self.time_step_sequence,
            "action_sequence": self.action_sequence,
        }

    def calculate_reward(self, done):
        reward = 0
        return 0, 0, 0

    def get_sample_number(self):
        return self.sample_number_count

    def render(self, mode="human", close=False):
        # This could visualize the current state if necessary
        pass