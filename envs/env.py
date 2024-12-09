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
from ddrm.functions.denoising import initialize_generalized_steps, denoise_single_step, efficient_generalized_steps
import pdb 
import lpips


"""
目前設計
1. ddrm 會預先在episode init的時候走完所有步數以作為未來比較的參考
2. ddrm 步數為uniform的discrete step
3. action space只能基於 ddrm的 discrete step 去選
4. 第一次的action space可以選ddrm_seq[0:end-4], 為了避免第一次的action就踩到沒得走五步
5. 後面的 action space 會隨著剩餘可以走的步數調整action
6. reward先暫定只根據lpips, 但同時可以計算PSNR, SSIM, LPIPS 

---
上面設計會有缺點，能挑選的步數比較少，如果只要有不小心跳太多步的就可能會不夠走
下個設計
第一步用discrete, 後面用continuous



"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionEnv(gym.Env):
    def __init__(self, runner, model, cls, target_steps=10, max_steps=100):
        super(DiffusionEnv, self).__init__()
        self.runner = runner
        self.target_steps = target_steps
        self.final_threshold = 0.9
        val_loader, sigma_0, config, deg, H_funcs, model, idx_so_far, cls_fn = self.runner.sample(cls)
        # pdb.set_trace()
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

        # Dataloader needs iterator!!!
        self.data_iter = iter(self.val_loader)
        self.GT_image, self.classes = next(self.data_iter)
        # print(np.size(self.GT_image))
        # pdb.set_trace()

        self.lpips_loss = lpips.LPIPS(net='vgg')

        # noise
        # current noise image is total noise, y_0 is origin image * H to generate degradation img
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

        # ddim sequence
        self.skip = self.runner.num_timesteps // self.runner.args.timesteps
        self.interval = self.runner.num_timesteps // target_steps
        # self.reversed_ddrm_seq = range(self.runner.num_timesteps - self.interval , 0, -1*self.skip)

        self.ddrm_seq = range(0, self.runner.num_timesteps, self.skip)
        self.ddrm_seq_reversed = self.ddrm_seq[::-1]
        # seq = range(0, self.runner.num_timesteps, skip)
        # seq_next = [-1] + list(seq[:-1])
        # self.ddrm_seq = list(reversed(seq))
        # self.ddrm_seq_next = list(reversed(seq_next))

        self.time_step_sequence = []
        self.action_sequence = []
        self.max_steps = max_steps
        self.current_step_num = 0


        # Define the action and observation space
        # 用Box的話會是連續的action space
        # self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(1,))

        # 第一步容許跳到離散數線上任意一步，而剩下的再由剩餘空間中找，但不能一次就走到最底，所以要減掉可走的步數
        # self.action_space = gym.spaces.Discrete(self.skip - self.max_steps)
        self.t = 999    # initialize the first timestep
        self.action_space = gym.spaces.Box(low=self.target_steps, high=self.t, shape=(1,))

        self.observation_space = Dict(
            {
                "image": Box(
                    low=0,
                    high=255,
                    # shape=(self.batch_size, 3*self.sample_size*self.sample_size),
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
        self.ddrm_current_image = self.current_noise_image.clone()

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(seed)
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        # print(f"Seed: {seed}")
        # return [seed]

    def reset(self, seed=None, options=None):
        self.episode_init = True
        if seed is not None:
            self.seed(seed)

        if self.current_image_idx < self.valdata_len-1:
            self.skip = self.runner.num_timesteps // self.runner.args.timesteps
            # print("Current image index: ", self.current_image_idx)
            # print("Val data length: ", self.valdata_len)
            self.current_image_idx += 1

            self.current_step_num = 0
            self.time_step_sequence = []
            self.action_sequence = []

            # if the sequence can reach x0, change to next image. 沒的話再用一次原本的圖片 (Q: 會不會有可能走不出來)
            if self.t == 0: 
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

        if self.episode_init:
            # FOR RL
            # Randomly pick last T in a ddrm sequence
            # last_T = torch.randint(low=min(self.ddrm_seq), high=max(self.ddrm_seq), size=(1,)).item()
            
            # 初始 T 先和DDRM的 T 對齊
            self.current_index = 0
            last_T = self.ddrm_seq[self.current_index]
            self.state = initialize_generalized_steps(
                self.current_noise_image.to("cuda"),
                last_T,
                self.runner.betas,
                self.H_funcs,
                self.y_0,
                self.sigma_0,
            )
            # self.t = next_t
            self.episode_init = False
            # print("type of self.t: ", type(self.t))

            # DDRM as intermediate GT
            self.ddrm_seq_result = self._perform_denoising_sequence(
                self.ddrm_current_image.to("cuda"),
                self.ddrm_seq
            )

        
        # add current t and current action

        # TODO: Action Design
        next_t = self._calculate_time_step(action)
        self.current_noise_image = self._perform_denoising_single_step(
            self.state, self.t, next_t
        )

        # Update t
        self.t = next_t
        self._update_sequences(self.t, action)


        # TODO done mechanism
        done = self.current_step_num == self.target_steps - 1
        # Calculate reward
        reward, lpips, ddrm_lpips, ssim, ddim_ssim = self.calculate_reward(done) 
        
        
        info = self._create_info_dict(self.t, reward, lpips, ddrm_lpips, ssim, ddim_ssim )
        observation = {"image": self.current_noise_image.squeeze(0), "value": self.t}

        
        self.current_step_num += 1
        # 下一次的action space會變小(只能從當前的 t 去往後挑)
        # self.skip = self.t // self.runner.args.timesteps
        # action space 選擇在剩餘還可以走的step上
        # self.action_space = gym.spaces.Discrete(self.skip - (self.max_steps - self.current_step_num))
        print("current_step_num: ", self.current_step_num)
        print("max_steps: ", self.max_steps)
        print("t = ", self.t)
        self.action_space = gym.spaces.Box(low=(self.target_steps - self.current_step_num), high=int(self.t), shape=(1,))

        return observation, reward, done, truncate, info

    def _calculate_time_step(self, action):
        action = action.item()
        # print("action: ", action)
        # print("interval: ", self.interval)
        # print("current_index: ", self.current_index)
        # print("current t: ", self.ddrm_seq_reversed[self.current_index])
        # t = int(torch.round(torch.tensor(self.ddrm_seq_reversed[self.current_index] - action)))
        t = int(torch.round(torch.tensor(action)))
        final_t = max(0, min(t, 999))
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
    
    # generate full DDRM
    def _perform_denoising_sequence(self, x, seq):
        with torch.no_grad():
            xs, x0_preds = efficient_generalized_steps(
                x,
                seq,
                self.model,
                self.runner.betas,
                self.H_funcs,
                self.y_0,
                self.sigma_0,
                etaB=self.runner.args.etaB,
                etaA=self.runner.args.eta,
                etaC=self.runner.args.eta,
                cls_fn=self.cls_fn,
                classes=self.classes,
            )

            x = torch.stack([inverse_data_transform(self.config, y.to(device)) for y in xs])
        return x

    def _create_info_dict(self, t, reward, lpips, ddrm_lpips, ssim, ddim_ssim ):
        return {
            # "ddim_t": ddim_t,
            "t": t,
            "reward": reward,
            "lpips": lpips,
            "ddrm_lpips": ddrm_lpips,
            "ssim": ssim,
            "ddim_ssim": ddim_ssim,
            "time_step_sequence": self.time_step_sequence,
            "action_sequence": self.action_sequence,
        }
    
    def calculate_reward(self, done):
        reward = 0

        # LPIPS loss 計算
        lpips_score = self.lpips_loss(
            ((self.current_noise_image[0] + 1.0) / 2.0).cpu(),
            ((self.GT_image[0] + 1.0) / 2.0).cpu(),
        ).item()

        ddim_lpips_score = self.lpips_loss(
            ((self.ddrm_current_image[0] + 1.0) / 2.0).cpu(),
            ((self.GT_image[0] + 1.0) / 2.0).cpu(),
        ).item()


        ssim = structural_similarity(
            ((self.current_noise_image[0] + 1.0) / 2.0).cpu().numpy(),
            ((self.GT_image[0] + 1.0) / 2.0).cpu().numpy(),
            multichannel=True,
            channel_axis=0,
            data_range=1,
        )
        ddim_ssim = structural_similarity(
            ((self.ddrm_current_image[0] + 1.0) / 2.0).cpu().numpy(),
            ((self.GT_image[0] + 1.0) / 2.0).cpu().numpy(),
            multichannel=True,
            channel_axis=0,
            data_range=1,
        )
        # LPIPS 越低表示影像越相似，因此獎勵應根據 score 越低越多
        # Intermediate reward
        if lpips_score < ddim_lpips_score:
            reward += 1 / self.target_steps

        # Sparse reward (LPIPS)
        if done and lpips_score < self.final_threshold:
            reward += 1

        return reward, lpips_score, ddim_lpips_score, ssim, ddim_ssim

    # def calculate_reward(self, done):
    #     reward = 0
    #     # similarity = torch.nn.functional.mse_loss(self.current_image, self.GT_image)
    #     # ddim_similarity = torch.nn.functional.mse_loss(self.ddrm_current_image, self.GT_image)
    #     ssim = structural_similarity(
    #         ((self.current_noise_image[0] + 1.0) / 2.0).cpu().numpy(),
    #         ((self.GT_image[0] + 1.0) / 2.0).cpu().numpy(),
    #         multichannel=True,
    #         channel_axis=0,
    #         data_range=1,
    #     )
    #     ddim_ssim = structural_similarity(
    #         ((self.ddrm_current_image[0] + 1.0) / 2.0).cpu().numpy(),
    #         ((self.GT_image[0] + 1.0) / 2.0).cpu().numpy(),
    #         multichannel=True,
    #         channel_axis=0,
    #         data_range=1,
    #     )
    #     # Intermediate reward
    #     if ssim > ddim_ssim:
    #         reward += 1 / self.target_steps
    #     # Sparse reward (SSIM)
    #     if done and ssim > self.final_threshold:
    #         reward += 1

    #     return reward, ssim, ddim_ssim

    def render(self, mode="human", close=False):
        # This could visualize the current state if necessary
        pass
