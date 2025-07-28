import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr 

import os
import sys
import glob
import multiprocessing
import random
import numpy as np
import logging
import yaml
import argparse
from PIL import Image
import shutil

from verti_bench.envs.terrain import TerrainManager
from verti_bench.vehicles.HMMWV import HMMWVManager
from verti_bench.rl.off_road_VertiBench_offlinerl import off_road_art
from stable_baselines3 import PPO

from verti_bench.envs.utils.utils import SetChronoDataDirectories

class IQLGym:
    def __init__(self, config):
        if config['use_gui'] and not config['render']:
            raise ValueError("If use_gui is True, render must also be True. GUI requires rendering.")
            
        # Store configuration parameters
        self.config = config
        self.world_id = config['world_id']
        if not (1 <= self.world_id <= 100):
            raise ValueError(f"World ID must be between 1 and 100, got {self.world_id}")
        self.scale_factor = config['scale_factor']
        self.render = config['render']
        self.use_gui = config['use_gui']
        self.vehicle_type = config['vehicle']
        self.system_type = config['system']
        self.max_time = config['max_time']
        self.speed = config['speed']
        
        supported_vehicles = ['hmmwv']
        if self.vehicle_type.lower() not in supported_vehicles:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}, "
                             f"only support {supported_vehicles[0]} for RL-based systems!")

        # RL specific attributes
        self.env = None
        self.model = None
        self.obs = None
        self.total_steps = 0
        
        # Clean up tmp terrain directory
        terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../envs/data/BenchMaps/sampled_maps/Configs/tmp")
        if os.path.exists(terrain_dir):
            shutil.rmtree(terrain_dir)
        
    def initialize(self):
        """Initialize the RL sim"""
        self.env = off_road_art(world_id=self.world_id, scale_factor=self.scale_factor)
        self.env.m_max_time = self.max_time
        self.env.max_speed = self.speed
        

        # ─── Load the IQL‐trained actor ─────────────────────────────────────────
        import torch
        import os
        from verti_bench.offlinerl_dev.algorithms.iql_vertibench import GaussianPolicy, DeterministicPolicy  # adjust import path

        # path to the saved IQL checkpoint (state_dict)
        # model_path = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)),
        #     "checkpoints",             # this should match your train config.checkpoints_path
        #     "YOUR_RUN_NAME",           # e.g. "IQL-VertiBench-world5-<uuid>"
        #     "checkpoint_100000.pt"     # or whatever final checkpoint you want
        # )
        model_path = "/home/zkr/Documents/verti_bench/checkpoints/IQL-VertiBench-world5-baae5c90/checkpoint_4999.pt"

        # load the checkpoint
        state = torch.load(model_path, map_location="cpu")

        # re-create the actor network
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        # if you trained with deterministic policy:
        # actor = DeterministicPolicy(obs_dim, act_dim, max_action)
        # otherwise (default) use GaussianPolicy:
        actor = GaussianPolicy(obs_dim, act_dim, max_action)

        actor.load_state_dict(state["actor"])
        actor.eval()           # set to eval mode for deterministic behavior
        self.model = actor
        print(f"Loaded IQL actor from {model_path}")
        
        # Reset the environment
        self.obs, self.info = self.env.reset()
        
        # Initialize visualization if rendering is enabled
        if self.render:
            self.env.render('follow')
        
        # Initialize tracking variables
        self.total_steps = 0

    def run(self):
        """Run the simulation"""
        if self.env is None or self.model is None:
            raise ValueError("Simulation not initialized. Call initialize() first.")
        
        self.total_steps = self.env.m_max_time / self.env.m_step_size
        
        for step in range(int(self.total_steps)):
            # action, _states = self.model.predict(self.obs, deterministic=True)
            action = self.model.act(self.obs)
            print(f"Step {step + 1}")
            print("Action: ", action)
            self.obs, reward, done, self.info = self.env.step(action)
            print("obs=", self.obs, "reward=", reward, "done=", done)
            if self.render:
                self.env.render('follow')
            if done:
                break
            
        # return info.get('time_to_goal'), info.get('success', False), info.get('roll_angles', []), info.get('pitch_angles', [])
        results = {
            'time_to_goal': self.info.get('time_to_goal', None),
            'success': self.info.get('success', False),
            'roll_angles': self.info.get('roll_angles', []),
            'pitch_angles': self.info.get('pitch_angles', []),
        }
        return results['time_to_goal'], results['success'], results['roll_angles'], results['pitch_angles']
    
def single_experiment(config):
    """Run a single experiment with the given configuration"""
    iql_gym = IQLGym(config)
    iql_gym.initialize()

    time_to_goal, success, roll_angles, pitch_angles = iql_gym.run()

    # Collect results
    result = {
        'time_to_goal': time_to_goal,
        'success': success,
        'roll_angles': roll_angles,
        'pitch_angles': pitch_angles,
        'avg_roll': np.mean(roll_angles) if roll_angles else None,
        'avg_pitch': np.mean(pitch_angles) if pitch_angles else None
    }
    
    return result

# if __name__ == "__main__":
#     # Load configuration file
#     SetChronoDataDirectories()

#     # Example configuration
#     config = {
#         'world_id': 75,
#         'scale_factor': 1.0,
#         'render': True,
#         'use_gui': False,
#         'vehicle': 'hmmwv',
#         'system': 'iql',
#         'max_time': 60.0,
#         'speed': 10.0,
#         'collect_trajectory': False,
#     }
    
#     # Run a single experiment
#     result = single_experiment(config)
#     print("Experiment Result:", result)

import json
import pandas as pd
import numpy as np
import contextlib

def multi_experiment(base_config,
                     runs_per_world=10,
                     config_json_path=None,
                     csv_path=None):
    """
    对指定的世界 ID 进行重复实验，并汇总结果。

    :param base_config: dict, 单次实验的基础配置字典（不含 world_id）
    :param runs_per_world: int, 每个 world_id 重复实验的次数
    :param config_json_path: str, 保存 low/mid/high id 列表的 JSON 的路径
    :param csv_path: str or None, 如果提供，则会把结果写到该 CSV 文件
    :return: pandas.DataFrame, 包含所有实验结果
    """
    # if config_json_path is None or not os.path.isfile(config_json_path):
    #     raise FileNotFoundError(f"Cannot find config JSON at {config_json_path}")

    # # 1) 读取 JSON
    # with open(config_json_path, 'r') as f:
    #     labels = json.load(f)

    # low_ids = labels['difficulty'].get('low', [])
    # mid_ids = labels['difficulty'].get('mid', [])
    # high_ids = labels['difficulty'].get('high', [])

    # world_ids = sorted(low_ids + mid_ids + high_ids)

    world_ids = [i for i in range(1, 101)]  # 假设我们要测试所有 100 个世界

    records = []
    for world_id in world_ids:
        for run_idx in range(runs_per_world):
            # 将 world_id 注入配置
            config = base_config.copy()
            config['world_id'] = world_id

            # 静默运行：屏蔽所有 stdout/stderr
            with open(os.devnull, 'w') as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                gym = IQLGym(config)
                gym.initialize()
                t2g, success, rolls, pitches = gym.run()
            
            # 这里是每次实验新定义一个Gym因而success的获取没问题
            records.append({
                'world_id':      world_id,
                'run_idx':       run_idx,
                'time_to_goal':  t2g,
                'success':       success,
                'avg_roll_deg':  np.mean(rolls) if rolls else np.nan,
                'avg_pitch_deg': np.mean(pitches) if pitches else np.nan,
            })

            # 可以选择保留进度打印
            print(f"[world {world_id} run {run_idx:02d}] time={t2g}, success={success}")

    df = pd.DataFrame(records)

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved all results to {csv_path}")

    return df


if __name__ == "__main__":
    SetChronoDataDirectories()
    base_config = {
        'scale_factor': 1.0,
        'render': False,
        'use_gui': False,
        'vehicle': 'hmmwv',
        'system': 'iql',
        'max_time': 60.0,
        'speed': 10.0,
        'collect_trajectory': False,   # 多实验时一般关闭数据收集，免得文件爆炸
    }

    df = multi_experiment(
        base_config,
        runs_per_world=5,
        config_json_path='/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final/config_ids.json',
        csv_path='iql_multi_experiment_results.csv'
    )
    print(df.groupby('world_id')['success'].mean())