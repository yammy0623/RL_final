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

class DiffusionEnv(gym.Env):
    def __init__(self, model_name, target_steps=10, max_steps=100):
        super(DiffusionEnv, self).__init__()
        self.target_steps = target_steps
        self.final_threshold = 0.9
        self._load_model_and_scheduler(model_name)
        self.sample_size = self.model.config.sample_size
        self.time_step_sequence = []
        self.action_sequence = []
        self.max_steps = max_steps
        self.current_step_num = 0
        self._define_action_and_observation_space()
        self.seed(232)
        self._initialize_images()
        self._generate_ground_truth_image()

    def _load_model_and_scheduler(self, model_name):
        if os.path.isdir(model_name):
            from diffusers_old import DDIMPipeline, DDIMScheduler, UNet2DModel
            print("Loading model from {}".format(model_name))
            subfolder = 'unet' if os.path.isdir(os.path.join(model_name, 'unet')) else None
            self.model = UNet2DModel.from_pretrained(model_name, subfolder=subfolder).eval()
            scheduler_subfolder = 'scheduler'
        else:
            from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
            print("Loading pretrained model from {}".format(model_name))
            self.model = UNet2DModel.from_pretrained(model_name).to("cuda")
            scheduler_subfolder = None

        self.model.to("cuda")
        self.scheduler = DDIMScheduler.from_pretrained(model_name, subfolder=scheduler_subfolder)
        self.scheduler.set_timesteps(self.max_steps)
        self.ddim_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder=scheduler_subfolder)
        self.ddim_scheduler.set_timesteps(self.target_steps)

    def _define_action_and_observation_space(self):
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,))
        self.observation_space = Dict({
            "image": Box(low=0, high=255, shape=(3, self.sample_size, self.sample_size), dtype=np.uint8),
            "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
        })

    def _initialize_images(self):
        self.current_image = torch.randn((1, 3, self.sample_size, self.sample_size), device="cuda", generator=self.generator)
        self.ddrm_current_image = self.current_image.clone()

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        self.generator = torch.Generator(device='cuda').manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        # print(f"Seed: {seed}")
        # return [seed]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._initialize_environment()
        self._generate_ground_truth_image()
        observation = self._create_initial_observation()
        return observation, {}

    def _initialize_environment(self):
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []
        self.current_image = torch.randn((1, 3, self.sample_size, self.sample_size), device="cuda", generator=self.generator)
        self.ddrm_current_image = self.current_image.clone()

    def _generate_ground_truth_image(self):
        input = self.current_image.clone().to("cuda")
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = self.model(input, t).sample
                prev_noisy_sample = self.scheduler.step(noisy_residual, t, input, generator=self.generator).prev_sample
                input = prev_noisy_sample
        self.GT_image = input.cpu()

    def _create_initial_observation(self):
        return {
            "image": self.current_image.cpu().numpy(),
            "value": np.array([999])
        }
    
    def step(self, action):
        truncate = self.current_step_num >= self.max_steps

        # Perform RL step
        t = self._calculate_time_step(action)
        self._update_sequences(t, action)
        self.current_image = self._perform_denoising_step(t, self.current_image)

        # Perform DDIM step
        ddim_t = self.ddim_scheduler.timesteps[self.current_step_num]
        self.ddrm_current_image = self._perform_denoising_step(ddim_t, self.ddrm_current_image)

        # Check if the episode is done
        done = self.current_step_num == self.target_steps - 1
        self.current_step_num += 1

        # Calculate reward
        reward, ssim, ddim_ssim = self.calculate_reward(done)
        info = self._create_info_dict(ddim_t, t, reward, ssim, ddim_ssim)

        observation = {
            "image": self.current_image,
            "value": t
        }

        return observation, reward, done, truncate, info

    def _calculate_time_step(self, action):
        interval = self.ddim_scheduler.timesteps[0] - self.ddim_scheduler.timesteps[1]
        ddim_t = self.ddim_scheduler.timesteps[self.current_step_num]
        t = int(torch.round(ddim_t - interval * action))
        return torch.tensor(max(0, min(t, 999)))

    def _update_sequences(self, t, action):
        self.time_step_sequence.append(t.item())
        self.action_sequence.append(action.item())

    def _perform_denoising_step(self, t, image):
        with torch.no_grad():
            input = image.to("cuda")
            noisy_residual = self.model(input, t).sample
            prev_noisy_sample = self.ddim_scheduler.step(noisy_residual, t, input, generator=self.generator).prev_sample
            return prev_noisy_sample.cpu()

    def _create_info_dict(self, ddim_t, t, reward, ssim, ddim_ssim):
        return {
            'ddim_t': ddim_t,
            't': t,
            'reward': reward,
            'ssim': ssim,
            'ddim_ssim': ddim_ssim,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence
        }
        # Save the image if done
        # if done:
        #     if not os.path.exists('img'):
        #         os.makedirs('img')
        #     images = (self.current_image / 2 + 0.5).clamp(0, 1)
        #     images = images.cpu().permute(0, 2, 3, 1).numpy()[0]
        #     images = Image.fromarray((images * 255).round().astype("uint8"))
        #     filename = os.path.join('img', f"RL_{self.current_step_num}.png")
        #     images.save(filename)
        #     images = (self.ddrm_current_image / 2 + 0.5).clamp(0, 1)
        #     images = images.cpu().permute(0, 2, 3, 1).numpy()[0]
        #     images = Image.fromarray((images * 255).round().astype("uint8"))
        #     filename = os.path.join('img', f"ddim_{self.current_step_num}.png")
        #     images.save(filename)
        #     images = (self.GT_image / 2 + 0.5).clamp(0, 1)
        #     images = images.cpu().permute(0, 2, 3, 1).numpy()[0]
        #     images = Image.fromarray((images * 255).round().astype("uint8"))
        #     filename = os.path.join('img', f"GT_{self.current_step_num}.png")
        #     images.save(filename)
       

    def calculate_reward(self, done):
        reward = 0
        # similarity = torch.nn.functional.mse_loss(self.current_image, self.GT_image)
        # ddim_similarity = torch.nn.functional.mse_loss(self.ddrm_current_image, self.GT_image)
        ssim = structural_similarity(((self.current_image[0]+1.0)/2.0).cpu().numpy(), ((self.GT_image[0]+1.0)/2.0).cpu().numpy() ,multichannel=True,channel_axis=0, data_range=1)
        ddim_ssim = structural_similarity(((self.ddrm_current_image[0]+1.0)/2.0).cpu().numpy(), ((self.GT_image[0]+1.0)/2.0).cpu().numpy() ,multichannel=True,channel_axis=0, data_range=1)
        # Intermediate reward
        if ssim > ddim_ssim:
            reward += 1/self.target_steps
        # Sparse reward (SSIM)
        if done and ssim > self.final_threshold:
            reward += 1

        return reward, ssim, ddim_ssim
    
    def render(self, mode='human', close=False):
        # This could visualize the current state if necessary
        pass