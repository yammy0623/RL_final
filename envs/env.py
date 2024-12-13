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
from ddrm.functions.denoising import initialize_generalized_steps, denoise_single_step, denoise_guided_addnoise
from skimage.metrics import structural_similarity
import pdb 

class DiffusionEnv(gym.Env):
    def __init__(self, runner, model, cls, target_steps=10, max_steps=100, agent1=None):
        super(DiffusionEnv, self).__init__()
        
        
        # Model
        self.last_T = 999
        self.runner = runner
        self.target_steps = target_steps
        self.final_threshold = 0.9
        val_loader, sigma_0, config, deg, H_funcs, model, idx_so_far, cls_fn = self.runner.sample(cls)
        self.val_loader = val_loader
        self.sigma_0 = sigma_0
        self.config = config
        self.deg = deg
        self.H_funcs = H_funcs
        self.model = model
        self.model.to("cuda")
        
        self.idx_so_far = idx_so_far
        self.cls_fn = cls_fn
        self.valdata_len = self.runner.val_datalen
        self.current_image_idx = 0
        self.sample_size = config.data.image_size
        self.batch_size = config.sampling.batch_size


        # RL Setting
        self.agent1 = agent1 # RL model from subtask 1
        self.target_steps = target_steps
        self.uniform_steps = [i for i in range(0, 999, 1000//self.target_steps)][::-1]    
        self.adjust = True if agent1 is not None else False    

        self.max_steps = max_steps

        # Count the number of steps
        self.current_step_num = 0 
        if self.adjust:
            self.action_space = gym.spaces.Box(low=-5, high=5)
        else:
            self.action_space = spaces.Discrete(20)

        # Define the action and observation space
        self.observation_space = Dict({
            "image": Box(low=-1, high=1, shape=(3, self.sample_size, self.sample_size), dtype=np.float32),
            "value": Box(low=np.array([0]), high=np.array([999]), dtype=np.uint16)
        })
    

        # ddim sequence
        skip = self.runner.num_timesteps // self.runner.args.timesteps
        self.interval = self.runner.num_timesteps // target_steps
        # seq = range(self.runner.num_timesteps, 0, -1*skip)
        seq = range(0, self.runner.num_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        self.ddim_seq = list(reversed(seq))
        self.ddim_seq_next = list(reversed(seq_next))

        self.time_step_sequence = []
        self.action_sequence = []
        self.max_steps = max_steps
        self.current_step_num = 0


        # Define the action and observation space
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,))
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
        self.seed(232)
        self.episode_init = True
        self.state = None
        # pdb.set_trace()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)

    def reset(self, seed=None, options=None):
        self.episode_init = True
        if seed is not None:
            self.seed(seed)

        # Reset counter, sequence
        self.current_step_num = 0
        self.time_step_sequence = []
        self.action_sequence = []

        # Load Image
        self.data_iter = iter(self.val_loader)
        self.GT_image, self.classes = next(self.data_iter)

        # noise and low level image y_0, 
        self.noise_image, self.y_0 = self.runner.sample_init(
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
        self.state = initialize_generalized_steps(
                self.noise_image.to("cuda"),
                self.last_T,
                self.runner.betas,
                self.H_funcs,
                self.y_0,
                self.sigma_0,
            )
        self.x0_t = self.state['x']

        self.ddim_state = initialize_generalized_steps(
            self.noise_image.to("cuda"),
            self.ddim_seq[0],
            self.runner.betas,
            self.H_funcs,
            self.y_0,
            self.sigma_0,
        )
        ddim_x0_t = self.ddim_state['x']

        # Precoputing DDRM uniform seq
        with torch.no_grad():
            for i in range(self.target_steps):
                ddim_t = torch.tensor(self.uniform_steps[i])
                ddim_x0_t, ddim_at, ddim_et = denoise_single_step(self.ddim_state, self.model, ddim_t, self.cls_fn, self.classes)
                if i != 0:
                    self.ddim_state['x'] = denoise_guided_addnoise(self.ddim_state, ddim_t, ddim_at, ddim_et, ddim_x0_t, self.H_funcs, self.sigma_0, self.runner.args)
        orig = inverse_data_transform(self.config, self.GT_image).to(self.runner.device)
        ddim_x = inverse_data_transform(self.config, ddim_x0_t).to(self.runner.device)
        ddim_mse = torch.mean((ddim_x - orig) ** 2)
        self.ddim_psnr = 10 * torch.log10(1 / ddim_mse).item()
        self.ddim_ssim = structural_similarity(ddim_x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        observation = {
            "image": torch.tensor(self.x0_t).squeeze(0).cpu().numpy(),
            "value": np.array([self.last_T]),
        }

        # TODO START HERE
        if self.adjust: # Second subtask
            action, _state = self.agent1.predict(observation, deterministic=True)
            self.action_sequence.append(action.item())

            start_t = 50 * (1+action) - 1
            next_t = torch.tensor(int(max(0, min(start_t, 999))))
            self.interval = int(next_t / (self.target_steps - 1)) 
            self.state['x'] = denoise_guided_addnoise(self.state, self.at, self.et, self.x0_t, self.H_funcs, self.sigma_0, self.runner.args)
            
            # Next round
            self.t = next_t
            self.x0_t, self.at, self.et = denoise_single_step(self.state, self.model, self.t, self.cls_fn, self.classes)
            self.time_step_sequence.append(self.t.item())
            observation = {
                    "image": self.x0_t.cpu(),
                    "value": np.array([self.t])
                }
            self.current_step_num += 1

        torch.cuda.empty_cache()  # Clear GPU cache
            
        return observation, {}

    def _load_next_image(self):
        self.GT_image, self.classes = next(self.data_iter)
        self.noise, self.y_0 = self.runner.sample_init(
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
        # RL
        with torch.no_grad():
            if self.adjust == False:
                start_t = 50 * (1+action) - 1
                next_t = torch.tensor(int(max(0, min(start_t, 999))))
                self.interval = int(next_t / (self.target_steps - 1)) 
                self.state['x'] = denoise_guided_addnoise(self.state, self.at, self.et, self.x0_t, self.H_funcs, self.sigma_0, self.runner.args)
                self.action_sequence.append(action.item())
            else: # Second subtask
                next_t = self.t - self.interval - self.interval * action
                next_t = torch.tensor(int(max(0, min(next_t, 999))))
                self.interval = int(next_t / (self.target_steps - self.current_step_num - 1)) if (self.target_steps - self.current_step_num - 1) != 0 else self.interval
                self.state['x'] = denoise_guided_addnoise(self.state, self.at, self.et, self.x0_t, self.H_funcs, self.sigma_0, self.runner.args)
                self.action_sequence.append(action.item())

        
        self.t = next_t
        self.x0_t, self.at, self.et = denoise_single_step(self.state, self.model, self.t, self.cls_fn, self.classes)
        self.time_step_sequence.append(self.t.item())

        # Run remaining steps via uniform policy
        self.uniform_x0_t = self.x0_t.clone()
        uniform_at = self.at.clone()
        uniform_et = self.et.clone()
        self.uniform_state = self.state.clone()

        for i in range(self.target_steps - self.current_step_num - 1): 
            uniform_t = torch.tensor(int(self.t - self.interval - self.interval * i))
            uniform_t = torch.tensor(max(0, min(uniform_t, 999)))
            self.uniform_state['x'] = denoise_guided_addnoise(self.state,  uniform_at, uniform_et, self.uniform_x0_t, self.H_funcs, self.sigma_0, self.runner.args)
            self.uniform_x0_t, uniform_at, uniform_et = denoise_single_step(self.uniform_state, self.model, uniform_t, self.cls_fn, self.classes)


        # Finish the episode if denoising is done
        done = (self.current_step_num == self.target_steps - 1) or not self.adjust
        reward, ssim, psnr, ddim_ssim, ddim_psnr = self.calculate_reward(done)
        info = {
            'ddim_t': self.uniform_steps[self.current_step_num],
            't': self.t,
            'reward': reward,
            'ssim': ssim,
            'psnr': psnr,
            'ddim_ssim': ddim_ssim,
            'ddim_psnr': ddim_psnr,
            'time_step_sequence': self.time_step_sequence,
            'action_sequence': self.action_sequence,
        }

        observation = {
            "image":  self.current_noise_image.cpu(),  
            "value": np.array([self.t])
        }
        self.current_step_num += 1
        torch.cuda.empty_cache()
        return observation, reward, done, truncate, info

    # def _calculate_time_step_first(self, action):
    #     start_t = 50 * (1 + action) - 1
    #     t = torch.tensor(int(max(0, min(start_t, self.last_T))))
    #     return t

    # def _calculate_time_step_second(self, action):
    #     t = self.t - self.interval - self.interval * action
    #     t = torch.tensor(int(max(0, min(t, self.last_T))))
    #     return t

    # def _update_sequences(self, t, action):
    #     self.time_step_sequence.append(t.item() if type(t) == torch.Tensor else t)
    #     self.action_sequence.append(action.item())

    def _perform_denoising_single_step(self, state, t, next_t):
        with torch.no_grad():
            x0_t, xt_next = denoise_single_step(
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
            # x = torch.stack([inverse_data_transform(self.config, y) for y in xs])
            # x = inverse_data_transform(self.config, x0_t)
        return x0_t

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
        orig = inverse_data_transform(self.config, self.GT_image).to(self.runner.device)
        if done and self.adjust:
            x = inverse_data_transform(self.config, self.x0_t).to(self.runner.device)
        else:
            x = inverse_data_transform(self.config, self.uniform_x0_t).to(self.runner.device)
        mse = torch.mean((x - orig) ** 2)
        psnr = 10 * torch.log10(1 / mse).item()
        ssim = structural_similarity(x.cpu().numpy(), orig.cpu().numpy(), win_size=21, channel_axis=0, data_range=1.0)
        
        # Intermediate reward (Percentage of temporary improvement)
        if not done and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            reward += 0.5/self.target_steps*psnr/self.ddim_psnr 
            reward += 0.5/self.target_steps*ssim/self.ddim_ssim
        
        # Sparse reward (Percentage of final improvement)
        if done and psnr > self.ddim_psnr and ssim > self.ddim_ssim:
            reward += 0.5*psnr/self.ddim_psnr
            reward += 0.5*ssim/self.ddim_ssim


        return reward, ssim, psnr, self.ddim_ssim, self.ddim_psnr

    def render(self, mode="human", close=False):
        # This could visualize the current state if necessary
        pass

    def set_adjust(self, adjust):
        self.adjust = adjust
        print(f"Set adjust to {adjust}")