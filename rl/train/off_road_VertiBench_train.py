import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from typing import Callable, List, Any, Optional, Sequence, Type
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import multiprocessing as mp
import torch as th
import numpy as np
import shutil

from verti_bench.rl.off_road_VertiBench import off_road_art
from verti_bench.rl.custom_networks.artCustomCNN import CustomCombinedExtractor

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.old_episode_num = 0
        self.old_timeout_count = 0
        self.old_fallen_count = 0
        self.old_success_count = 0
        self.old_crash_count = 0
        self.last_ep_rew_mean = 0.0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,  # 算法名称
            "learning_rate": self.model.learning_rate,   # 学习率
            "gamma": self.model.gamma,                   # 折扣因子
            "n_steps": self.model.n_steps,               # 每次更新的步数
            "batch_size": self.model.batch_size,         # 批处理大小
            "gae_lambda": self.model.gae_lambda,         # GAE lambda参数
            "clip_range": self.model.clip_range,         # PPO裁剪范围
            "ent_coef": self.model.ent_coef,             # 熵系数
            "vf_coef": self.model.vf_coef,               # 价值函数系数
            "max_grad_norm": self.model.max_grad_norm,   # 最大梯度裁剪
            "n_epochs": self.model.n_epochs,             # 每次更新的训练轮数
        }
        metric_dict = {
            "rollout/ep_rew_mean": 0,        # 平均回合奖励
            "rollout/ep_len_mean": 0,        # 平均回合长度
            "train/value_loss": 0.0,         # 价值函数损失
            "train/policy_loss": 0.0,        # 策略损失
            "train/entropy_loss": 0.0,       # 熵损失
            "train/approx_kl": 0.0,          # 近似KL散度
            "train/clip_fraction": 0.0,      # 裁剪比例
            "train/explained_variance": 0.0, # 解释方差
            "rollout/total_success": 0,      # 成功总次数
            "rollout/total_fallen": 0,       # 跌倒总次数
            "rollout/total_timeout": 0,      # 超时总次数
            "rollout/total_crashes": 0,      # 碰撞总次数
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        return True
    
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        Aggregate data from all environments
        """
        total_success_count = sum(self.training_env.get_attr("m_success_count"))
        total_fallen_count = sum(self.training_env.get_attr("m_fallen_count"))
        total_timeout_count = sum(self.training_env.get_attr("m_timeout_count"))
        total_episode_num = sum(self.training_env.get_attr("m_episode_num"))
        total_crash_count = sum(self.training_env.get_attr("m_crash_count"))

        # Log the rates
        self.logger.record("rollout/total_success", total_success_count)
        self.logger.record("rollout/total_fallen", total_fallen_count)
        self.logger.record("rollout/total_timeout", total_timeout_count)
        self.logger.record("rollout/total_episode_num", total_episode_num)
        self.logger.record("rollout/total_crashes", total_crash_count)
        
        self.old_episode_num = total_episode_num
        self.old_timeout_count = total_timeout_count
        self.old_fallen_count = total_fallen_count
        self.old_success_count = total_success_count
        self.old_crash_count = total_crash_count
        
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
            self.last_ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

        return True

    def _on_training_end(self) -> None:
        print("Training ended")
        print("Total episodes ran: ", self.old_episode_num)
        print("Total success count: ", self.old_success_count)
        print("Total fallen count: ", self.old_fallen_count)
        print("Total timeout count: ", self.old_timeout_count)
        print("Total crash count: ", self.old_crash_count)
        return True

def make_env(rank: int = 0, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param stage: (int) the terrain stage
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = off_road_art(world_id=1, scale_factor=1.0)
        env.reset(seed=seed + rank)
        return env
        
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../envs/data/BenchMaps/sampled_maps/Configs/tmp")
    if os.path.exists(terrain_dir):
        shutil.rmtree(terrain_dir)
    
    # Maximum num is 32
    num_procs = 5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    base_log_path = "./res/logs"
    os.makedirs(base_log_path , exist_ok=True)

    n_steps = 12000
    num_updates = 15
    timesteps_per_iteration = num_updates * n_steps * num_procs
    
    checkpoint_dir = './res/models'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Vectorized environment
    env = make_vec_env(env_id=make_env(), n_envs=num_procs, vec_env_cls=SubprocVecEnv)
    
    policy_kwargs = dict(
        # features_extractor_class=CustomCombinedExtractor,
        # features_extractor_kwargs={'features_dim': 32},
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    new_logger = configure(base_log_path, ["stdout", "csv", "tensorboard"])
    model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=n_steps, batch_size=n_steps // 2, 
                policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=base_log_path, device=device)
    print(model.policy)
    model.set_logger(new_logger)   

    # If training is interrupted, set the checkpoint file and starting index
    start_idx = 10
    checkpoint_path = os.path.join(checkpoint_dir, f"ppo_checkpoint{start_idx}")
    if os.path.exists(checkpoint_path + ".zip"):
        model = PPO.load(checkpoint_path, env)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        start_idx = 0  # If no checkpoint, start from scratch

    num_iterations = 40
    for i in range(start_idx, num_iterations):  
        model.learn(timesteps_per_iteration, progress_bar=True, callback=TensorboardCallback())
        # 保存完整模型（包含所有信息）
        model.save(os.path.join(checkpoint_dir, f"ppo_checkpoint{i}"))
        # 保存策略网络权重（仅权重参数）
        th.save(model.policy.state_dict(), os.path.join(checkpoint_dir, f"ppo_checkpoint{i}.pt"))
        print(f"Saved checkpoint {i}")

    print("Training completed!")
    