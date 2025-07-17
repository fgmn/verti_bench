#!/usr/bin/env python3
"""
轨迹数据处理脚本：将HDF5轨迹文件转换为离线强化学习所需的格式

将原始轨迹数据转换为 (状态, 动作, 奖励, 下一个状态, 下一个动作) 元组形式
其中奖励根据 off_road_VertiBench_offlinerl.py 中的 get_reward 函数计算
"""

import os
import sys
import h5py
import numpy as np
import pickle
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Any
import glob
import torch
import torch.nn.functional as F
import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入SWAE模型和相关工具
try:
    from envs.utils.swae_model import SWAE
    from rl.custom_networks.swae_model import LatentSpaceMapper
    SWAE_AVAILABLE = True
    print("✓ SWAE模型导入成功")
except ImportError as e:
    print(f"⚠ SWAE模型导入失败: {e}")
    print("将使用简化的地形特征提取")
    SWAE_AVAILABLE = False

class TrajectoryProcessor:
    """轨迹数据处理器"""
    
    def __init__(self, device='cpu'):
        self.trajs_dir = "/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Trajs_HDF5"
        self.device = torch.device(device)
        
        # 奖励计算参数（来自 get_reward 函数）
        self.progress_scale = 50.0  # 进度奖励系数
        self.roll_threshold = np.radians(30)   # 横滚角阈值 [rad]
        self.pitch_threshold = np.radians(30)  # 俯仰角阈值 [rad]
        self.roll_penalty_scale = 20.0   # 横滚惩罚系数
        self.pitch_penalty_scale = 20.0  # 俯仰惩罚系数
        self.stagnant_penalty = 10.0     # 停滞惩罚
        self.min_progress = 0.01         # 最小进度阈值 [m]
        
        # 地形特征参数
        self.patch_size = 64  # 地形patch尺寸
        self.max_speed = 4.0  # 最大速度
        
        # 初始化SWAE模型
        self.swae = None
        self.latent_space_mapper = None
        self.min_vector = None
        self.max_vector = None
        self.use_swae = False
        
        if SWAE_AVAILABLE:
            self._load_swae_model()
    
    def _load_swae_model(self):
        """加载SWAE模型和相关组件"""
        try:
            # 模型文件路径
            utils_dir = "/home/zkr/Documents/verti_bench/envs/utils"
            swae_model_path = os.path.join(utils_dir, "BenchElev.pth")
            min_vector_path = os.path.join(utils_dir, "min_vectorBench.npy")
            max_vector_path = os.path.join(utils_dir, "max_vectorBench.npy")
            
            # 检查文件是否存在
            if not all(os.path.exists(p) for p in [swae_model_path, min_vector_path, max_vector_path]):
                print(f"⚠ SWAE模型文件缺失，将使用统计特征")
                return
            
            # 加载SWAE模型
            self.swae = SWAE(in_channels=1, latent_dim=64).to(self.device)
            self.swae.load_state_dict(torch.load(swae_model_path, map_location=self.device, weights_only=True))
            self.swae.freeze_encoder()
            self.swae.eval()
            
            # 加载归一化向量
            self.min_vector = torch.tensor(np.load(min_vector_path), dtype=torch.float32).to(self.device)
            self.max_vector = torch.tensor(np.load(max_vector_path), dtype=torch.float32).to(self.device)
            
            # 获取潜在空间映射器 (64维 -> 16维)
            self.features_dim = 16
            self.latent_space_mapper = LatentSpaceMapper(64, self.features_dim).to(self.device)
            
            self.use_swae = True
            print(f"✓ SWAE模型加载成功，设备: {self.device}")
            print(f"  - 模型将提取16维地形特征")
            
        except Exception as e:
            print(f"⚠ SWAE模型加载失败: {e}")
            print("将使用统计特征作为地形描述")
            self.use_swae = False
        
    def find_trajectory_files(self) -> List[str]:
        """查找所有HDF5轨迹文件"""
        pattern = os.path.join(self.trajs_dir, "**", "*.h5")
        files = glob.glob(pattern, recursive=True)
        return sorted(files)
    
    def load_trajectory_data(self, hdf5_path: str) -> Dict[str, Any]:
        """从HDF5文件加载轨迹数据"""
        print(f"Loading trajectory from: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # 加载元数据
            metadata = {}
            if 'metadata' in f:
                for key, value in f['metadata'].attrs.items():
                    metadata[key] = value
            
            # 加载状态数据
            states = {}
            if 'states' in f:
                for key in f['states'].keys():
                    states[key] = f['states'][key][:]
            
            # 加载动作数据
            actions = {}
            if 'actions' in f:
                for key in f['actions'].keys():
                    actions[key] = f['actions'][key][:]
            
            # 加载局部地形数据（如果存在）
            local_terrain = {}
            if 'local_terrain' in f:
                for key in f['local_terrain'].keys():
                    local_terrain[key] = f['local_terrain'][key][:]
        
        return {
            'metadata': metadata,
            'states': states,
            'actions': actions,
            'local_terrain': local_terrain
        }
    
    def compute_reward(self, state_curr: Dict, state_next: Dict, progress: float) -> float:
        """
        根据 get_reward 函数计算奖励
        
        Args:
            state_curr: 当前状态
            state_next: 下一个状态  
            progress: 进度 (m) = 上一步到目标距离 - 当前到目标距离
            
        Returns:
            reward: 计算得到的奖励值
        """
        # 基础进度奖励
        reward = self.progress_scale * progress
        
        # 停滞惩罚
        if np.abs(progress) < self.min_progress:
            reward -= self.stagnant_penalty
        
        # 获取车辆姿态角度（从 orientation 中获取 roll, pitch）
        orientation = state_curr['orientation']  # [roll, pitch, yaw]
        roll = orientation[0]   # 横滚角 [rad]
        pitch = orientation[1]  # 俯仰角 [rad]
        
        # 横滚角惩罚
        if abs(roll) > self.roll_threshold:
            roll_penalty_scale = self.roll_penalty_scale * np.abs(roll / self.roll_threshold)
            reward -= roll_penalty_scale * (abs(roll) - self.roll_threshold)
        
        # 俯仰角惩罚
        if abs(pitch) > self.pitch_threshold:
            pitch_penalty_scale = self.pitch_penalty_scale * np.abs(pitch / self.pitch_threshold)
            reward -= pitch_penalty_scale * (abs(pitch) - self.pitch_threshold)
        
        return reward
    
    def create_observation_vector(self, state: Dict, local_terrain: Dict, idx: int) -> np.ndarray:
        """
        创建观测向量（模拟 get_observation 函数的输出）
        
        根据 off_road_VertiBench_offlinerl.py，观测向量包含：
        - 16维地形特征（SWAE编码的局部地形）
        - 1维航向误差（归一化到[-1,1]）
        - 1维归一化速度
        
        总共18维观测向量
        """
        # 获取地形特征（如果有局部地形数据）
        if 'elevation_patches' in local_terrain and local_terrain['elevation_patches'].shape[0] > idx:
            elevation_patch = local_terrain['elevation_patches'][idx]  # 64x64
            
            if self.use_swae and self.swae is not None:
                # 使用SWAE模型提取16维地形特征
                terrain_features = self._extract_swae_features(elevation_patch)
            else:
                # 简化的地形特征提取（统计特征）
                terrain_features = self._extract_statistical_features(elevation_patch)
            
        else:
            # 如果没有地形数据，使用零向量
            terrain_features = np.zeros(16, dtype=np.float32)
        
        # 计算航向误差（简化版本，假设local_goal指向目标）
        if 'local_goal' in state and state['local_goal'].size >= 2:
            # 获取车辆位置和目标位置
            vehicle_pos = state['position'][:2]  # [x, y]
            local_goal = state['local_goal']     # [x, y]
            
            # 计算到目标的方向
            goal_vector = local_goal - vehicle_pos
            if np.linalg.norm(goal_vector) > 1e-6:
                goal_heading = np.arctan2(goal_vector[1], goal_vector[0])
            else:
                goal_heading = 0.0
            
            # 获取车辆朝向
            vehicle_heading = state['orientation'][2]  # yaw角
            
            # 计算航向误差
            heading_error = (goal_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
            normalized_heading_diff = heading_error / np.pi  # 归一化到[-1, 1]
        else:
            normalized_heading_diff = 0.0
        
        # 计算归一化速度
        velocity = state['velocity']  # [vx, vy, vz]
        vehicle_speed = np.linalg.norm(velocity)
        max_speed = 4.0  # 假设最大速度为4 m/s
        normalized_speed = np.clip(vehicle_speed / max_speed, -1.0, 1.0)
        
        # 组合观测向量：16维地形特征 + 1维航向误差 + 1维速度
        observation = np.concatenate([
            terrain_features,                               # 16维地形特征
            [normalized_heading_diff],                      # 1维航向误差
            [normalized_speed]                              # 1维归一化速度
        ]).astype(np.float32)
        
        return observation
    
    def create_action_vector(self, action: Dict) -> np.ndarray:
        """
        创建动作向量
        
        根据动作空间，包含：
        - steering: 转向输入 [-1, 1]
        - throttle: 油门输入 [0, 1] 
        - braking: 制动输入 [0, 1]
        
        实际使用steering和根据throttle/braking计算的归一化速度
        """
        steering = action['steering']
        throttle = action['throttle'] 
        braking = action['braking']
        
        # 将油门和制动合并为一个速度控制信号
        # 正值表示加速，负值表示制动
        speed_control = throttle - braking
        
        # 动作向量：[转向, 速度控制]
        action_vector = np.array([steering, speed_control], dtype=np.float32)
        
        return action_vector
    
    def process_single_trajectory(self, hdf5_path: str) -> List[Tuple]:
        """
        处理单条轨迹，生成(s, a, r, s', a', done)元组序列
        
        对于最后一帧，使用占位符作为下一个状态和动作，并将done设置为True
        这确保了每个时间步都有对应的transition，保持数据完整性
        
        Returns:
            List of tuples: [(state, action, reward, next_state, next_action, done), ...]
        """
        # 加载数据
        data = self.load_trajectory_data(hdf5_path)
        
        states = data['states']
        actions = data['actions'] 
        local_terrain = data['local_terrain']
        metadata = data['metadata']
        
        # 获取轨迹长度
        num_timesteps = len(states['timestep'])
        
        if num_timesteps < 1:
            print(f"轨迹过短: {num_timesteps} 个时间步")
            return []
        
        print(f"处理轨迹，时间步数: {num_timesteps}")
        print(f"将生成 {num_timesteps} 个转换（包括最后一帧的占位符转换）")
        
        # 生成元组序列
        tuples = []
        
        for i in range(num_timesteps):
            # 构建当前状态
            state_curr = {
                'position': states['position'][i],           # 位置 [m]
                'orientation': states['orientation'][i],     # 姿态 [rad] 
                'velocity': states['velocity'][i],           # 速度 [m/s]
                'angular_velocity': states['angular_velocity'][i], # 角速度 [rad/s]
                'distance_to_goal': states['distance_to_goal'][i], # 到目标距离 [m]
                'local_goal': states['local_goal'][i],       # 局部目标 [m]
                'ground_height': states['ground_height'][i], # 地面高度 [m]
                'terrain_normal': states['terrain_normal'][i], # 地形法向量
                'slope_angle': states['slope_angle'][i]      # 坡度角 [rad]
            }
            
            # 构建当前动作
            action_curr = {
                'steering': actions['steering'][i],          # 转向输入
                'throttle': actions['throttle'][i],          # 油门输入
                'braking': actions['braking'][i],            # 制动输入
                'target_speed': actions['target_speed'][i]   # 目标速度
            }
            
            # 判断是否为最后一帧
            if i == num_timesteps - 1:
                # 最后一帧：使用占位符作为下一个状态和动作
                state_next = state_curr.copy()  # 下一个状态使用当前状态作为占位符
                action_next = action_curr.copy()  # 下一个动作使用当前动作作为占位符
                done = True  # 最后一帧必须标记为终止
                progress = 0.0  # 最后一帧没有进度
            else:
                # 非最后一帧：正常构建下一个状态和动作
                state_next = {
                    'position': states['position'][i+1],
                    'orientation': states['orientation'][i+1],
                    'velocity': states['velocity'][i+1],
                    'angular_velocity': states['angular_velocity'][i+1],
                    'distance_to_goal': states['distance_to_goal'][i+1],
                    'local_goal': states['local_goal'][i+1],
                    'ground_height': states['ground_height'][i+1],
                    'terrain_normal': states['terrain_normal'][i+1],
                    'slope_angle': states['slope_angle'][i+1]
                }
                
                action_next = {
                    'steering': actions['steering'][i+1],
                    'throttle': actions['throttle'][i+1],
                    'braking': actions['braking'][i+1],
                    'target_speed': actions['target_speed'][i+1]
                }
                
                # 计算进度
                distance_curr = state_curr['distance_to_goal']
                distance_next = state_next['distance_to_goal']
                progress = distance_curr - distance_next  # 进度 = 距离减少量
                
                # 判断是否提前终止（到达目标）
                done = distance_next < 1.0  # 到达目标附近 (距离小于1米)
            
            # 计算奖励
            reward = self.compute_reward(state_curr, state_next, progress)
            
            # 创建观测向量
            obs_curr = self.create_observation_vector(state_curr, local_terrain, i)
            
            # 对于最后一帧，下一个观测使用当前观测作为占位符
            if i == num_timesteps - 1:
                obs_next = obs_curr.copy()
            else:
                obs_next = self.create_observation_vector(state_next, local_terrain, i+1)
            
            # 创建动作向量
            act_curr = self.create_action_vector(action_curr)
            act_next = self.create_action_vector(action_next)
            
            # 创建元组 (s, a, r, s', a', done)
            transition = (obs_curr, act_curr, reward, obs_next, act_next, done)
            tuples.append(transition)
        
        print(f"生成 {len(tuples)} 个转换元组")
        return tuples
    
    def save_processed_data(self, tuples: List[Tuple], output_path: str, metadata: Dict = None):
        """保存处理后的数据"""
        data = {
            'transitions': tuples,
            'metadata': metadata or {},
            'format': 'offline_rl_transitions',
            'description': 'Processed trajectory data for offline RL: (state, action, reward, next_state, next_action, done)',
            'observation_dim': 18,  # 16地形 + 1航向 + 1速度
            'action_dim': 2,        # 转向 + 速度控制
        }
        
        # 添加统计信息
        if tuples:
            rewards = [t[2] for t in tuples]
            dones = [t[5] for t in tuples]
            data['statistics'] = {
                'num_transitions': len(tuples),
                'num_episodes': sum(dones),  # 统计episode数量
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards)
            }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"保存处理后的数据到: {output_path}")
        
        # 打印统计信息
        if 'statistics' in data:
            stats = data['statistics']
            print(f"统计信息:")
            print(f"  转换数量: {stats['num_transitions']}")
            print(f"  Episode数量: {stats['num_episodes']}")
            print(f"  奖励 - 平均: {stats['reward_mean']:.3f}, 标准差: {stats['reward_std']:.3f}")
            print(f"  奖励 - 最小: {stats['reward_min']:.3f}, 最大: {stats['reward_max']:.3f}")
    
    def _extract_swae_features(self, elevation_patch: np.ndarray) -> np.ndarray:
        """
        使用SWAE模型提取地形特征
        
        Args:
            elevation_patch: 64x64高度图patch
            
        Returns:
            16维SWAE特征向量
        """
        try:
            # 估算地形高度范围进行归一化（基于RL环境的处理）
            min_height = 0.0  # 假设最小高度
            max_height = 10.0  # 假设最大高度范围
            
            # 归一化到[-1, 1]
            normalized_patch = 2 * (elevation_patch - min_height) / (max_height - min_height) - 1
            normalized_patch = np.clip(normalized_patch, -1.0, 1.0)
            
            # 转换为tensor，添加batch和channel维度
            patch_tensor = torch.tensor(normalized_patch, dtype=torch.float32).unsqueeze(0).to(self.device)
            patch_tensor = patch_tensor.view(-1, 1, self.patch_size, self.patch_size)
            
            # SWAE特征提取
            with torch.no_grad():
                _, _, z = self.swae(patch_tensor)  # 输出64维潜在特征
                
                # 归一化潜在特征
                z_normalized = 2 * (z - self.min_vector) / (self.max_vector - self.min_vector) - 1
                z_normalized = torch.clamp(z_normalized, -1.0, 1.0)
                
                # 映射到16维
                mapped_features = self.latent_space_mapper(z_normalized)  # 64维 -> 16维
                
                return mapped_features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"⚠ SWAE特征提取失败: {e}, 使用统计特征")
            return self._extract_statistical_features(elevation_patch)
    
    def _extract_statistical_features(self, elevation_patch: np.ndarray) -> np.ndarray:
        """
        提取统计地形特征（备用方案）
        
        Args:
            elevation_patch: 64x64高度图patch
            
        Returns:
            16维统计特征向量
        """
        # 计算16个统计特征作为地形描述
        terrain_features = np.array([
            np.mean(elevation_patch),           # 平均高度
            np.std(elevation_patch),            # 高度标准差
            np.min(elevation_patch),            # 最小高度
            np.max(elevation_patch),            # 最大高度
            np.median(elevation_patch),         # 中位数高度
            np.percentile(elevation_patch, 25), # 25%分位数
            np.percentile(elevation_patch, 75), # 75%分位数
            np.var(elevation_patch),            # 方差
            # 梯度特征
            np.mean(np.gradient(elevation_patch, axis=0)),  # x方向平均梯度
            np.mean(np.gradient(elevation_patch, axis=1)),  # y方向平均梯度
            np.std(np.gradient(elevation_patch, axis=0)),   # x方向梯度标准差
            np.std(np.gradient(elevation_patch, axis=1)),   # y方向梯度标准差
            # 纹理特征
            np.mean(np.abs(np.diff(elevation_patch, axis=0))),  # x方向粗糙度
            np.mean(np.abs(np.diff(elevation_patch, axis=1))),  # y方向粗糙度
            np.sum(elevation_patch > np.mean(elevation_patch)) / elevation_patch.size,  # 高于平均的比例
            np.sum(elevation_patch < np.mean(elevation_patch)) / elevation_patch.size   # 低于平均的比例
        ])
        
        # 归一化地形特征到[-1, 1]范围
        terrain_features = np.clip(terrain_features, -10, 10)  # 先裁剪极值
        terrain_features = terrain_features / 10.0             # 归一化
        
        return terrain_features.astype(np.float32)

class OfflineRLDatasetCreator:
    """创建离线RL数据集的HDF5文件管理器"""
    
    def __init__(self, output_dir: str, transitions_per_file: int = 100000):
        """
        初始化数据集创建器
        
        Args:
            output_dir: 输出目录
            transitions_per_file: 每个文件的转换数量
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.transitions_per_file = transitions_per_file
        self.current_file_index = 0
        self.current_file = None
        self.current_file_path = None
        self.current_transitions_count = 0
        
        # 数据缓冲区
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.next_actions_buffer = []
        self.dones_buffer = []  # 添加done标志缓冲区
        
        # 统计信息
        self.total_transitions = 0
        self.total_files = 0
        self.metadata = {}
        
    def _create_new_file(self):
        """创建新的HDF5文件"""
        if self.current_file is not None:
            self.current_file.close()
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"offline_rl_dataset_{self.current_file_index:03d}_{timestamp}.h5"
        self.current_file_path = self.output_dir / filename
        
        # 创建HDF5文件
        self.current_file = h5py.File(self.current_file_path, 'w')
        
        # 创建元数据组
        meta_group = self.current_file.create_group('metadata')
        meta_group.attrs['file_index'] = self.current_file_index
        meta_group.attrs['max_transitions'] = self.transitions_per_file
        meta_group.attrs['creation_time'] = timestamp
        meta_group.attrs['format'] = 'offline_rl_transitions'
        meta_group.attrs['description'] = 'Offline RL dataset: (state, action, reward, next_state, next_action, done)'
        
        # 添加处理器元数据
        for key, value in self.metadata.items():
            meta_group.attrs[key] = value
            
        # 创建数据组
        data_group = self.current_file.create_group('transitions')
        
        # 创建数据集（可扩展）
        max_size = self.transitions_per_file
        
        # 状态数据集 (假设18维观测)
        data_group.create_dataset('states', 
                                 (0, 18), 
                                 maxshape=(max_size, 18), 
                                 dtype='f4',
                                 chunks=True,
                                 compression='gzip')
        
        # 动作数据集 (假设2维动作)
        data_group.create_dataset('actions', 
                                 (0, 2), 
                                 maxshape=(max_size, 2), 
                                 dtype='f4',
                                 chunks=True,
                                 compression='gzip')
        
        # 奖励数据集
        data_group.create_dataset('rewards', 
                                 (0,), 
                                 maxshape=(max_size,), 
                                 dtype='f4',
                                 chunks=True,
                                 compression='gzip')
        
        # 下一状态数据集
        data_group.create_dataset('next_states', 
                                 (0, 18), 
                                 maxshape=(max_size, 18), 
                                 dtype='f4',
                                 chunks=True,
                                 compression='gzip')
        
        # 下一动作数据集
        data_group.create_dataset('next_actions', 
                                 (0, 2), 
                                 maxshape=(max_size, 2), 
                                 dtype='f4',
                                 chunks=True,
                                 compression='gzip')
        
        # done标志数据集
        data_group.create_dataset('dones', 
                                 (0,), 
                                 maxshape=(max_size,), 
                                 dtype='bool',
                                 chunks=True,
                                 compression='gzip')
        
        self.current_transitions_count = 0
        self.total_files += 1
        
        print(f"创建新的HDF5文件: {self.current_file_path}")
        
    def add_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                      next_state: np.ndarray, next_action: np.ndarray, done: bool):
        """添加单个转换"""
        self.states_buffer.append(state)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)
        self.next_states_buffer.append(next_state)
        self.next_actions_buffer.append(next_action)
        self.dones_buffer.append(done)
        
        # 如果缓冲区达到一定大小，flush到文件
        if len(self.states_buffer) >= 1000:  # 每1000个转换flush一次
            self._flush_buffer()
            
    def add_transitions_batch(self, transitions: List[Tuple]):
        """批量添加转换"""
        for state, action, reward, next_state, next_action, done in transitions:
            self.add_transition(state, action, reward, next_state, next_action, done)
            
    def _flush_buffer(self):
        """将缓冲区数据写入当前文件"""
        if not self.states_buffer:
            return
            
        while self.states_buffer:  # 循环处理直到缓冲区为空
            # 检查是否需要创建新文件
            if self.current_file is None:
                self._create_new_file()
            
            # 计算当前文件还能容纳多少个转换
            remaining_capacity = self.transitions_per_file - self.current_transitions_count
            
            if remaining_capacity <= 0:
                # 当前文件已满，完成并创建新文件
                self._finalize_current_file()
                self.current_file_index += 1
                self._create_new_file()
                remaining_capacity = self.transitions_per_file
            
            # 确定本次写入的数量
            write_count = min(len(self.states_buffer), remaining_capacity)
            
            # 写入数据
            self._write_buffer_chunk(write_count)
            
    def _write_buffer_chunk(self, count: int):
        """写入缓冲区的一部分数据到当前文件"""
        if count <= 0 or not self.states_buffer:
            return
            
        start_idx = self.current_transitions_count
        end_idx = start_idx + count
        
        # 扩展数据集
        data_group = self.current_file['transitions']
        data_group['states'].resize((end_idx, 18))
        data_group['actions'].resize((end_idx, 2))
        data_group['rewards'].resize((end_idx,))
        data_group['next_states'].resize((end_idx, 18))
        data_group['next_actions'].resize((end_idx, 2))
        data_group['dones'].resize((end_idx,))
        
        # 写入数据
        data_group['states'][start_idx:end_idx] = np.array(self.states_buffer[:count])
        data_group['actions'][start_idx:end_idx] = np.array(self.actions_buffer[:count])
        data_group['rewards'][start_idx:end_idx] = np.array(self.rewards_buffer[:count])
        data_group['next_states'][start_idx:end_idx] = np.array(self.next_states_buffer[:count])
        data_group['next_actions'][start_idx:end_idx] = np.array(self.next_actions_buffer[:count])
        data_group['dones'][start_idx:end_idx] = np.array(self.dones_buffer[:count])
        
        # 更新计数
        self.current_transitions_count += count
        self.total_transitions += count
        
        # 从缓冲区移除已写入的数据
        self.states_buffer = self.states_buffer[count:]
        self.actions_buffer = self.actions_buffer[count:]
        self.rewards_buffer = self.rewards_buffer[count:]
        self.next_states_buffer = self.next_states_buffer[count:]
        self.next_actions_buffer = self.next_actions_buffer[count:]
        self.dones_buffer = self.dones_buffer[count:]
        
        print(f"写入 {count} 个转换到文件 {self.current_file_index} "
              f"(总数: {self.current_transitions_count}/{self.transitions_per_file})")
    
    def _write_full_buffer(self):
        """写入缓冲区的全部数据到当前文件"""
        if not self.states_buffer:
            return
            
        buffer_size = len(self.states_buffer)
        start_idx = self.current_transitions_count
        end_idx = start_idx + buffer_size
        
        # 扩展数据集
        data_group = self.current_file['transitions']
        data_group['states'].resize((end_idx, 18))
        data_group['actions'].resize((end_idx, 2))
        data_group['rewards'].resize((end_idx,))
        data_group['next_states'].resize((end_idx, 18))
        data_group['next_actions'].resize((end_idx, 2))
        data_group['dones'].resize((end_idx,))
        
        # 写入数据
        data_group['states'][start_idx:end_idx] = np.array(self.states_buffer)
        data_group['actions'][start_idx:end_idx] = np.array(self.actions_buffer)
        data_group['rewards'][start_idx:end_idx] = np.array(self.rewards_buffer)
        data_group['next_states'][start_idx:end_idx] = np.array(self.next_states_buffer)
        data_group['next_actions'][start_idx:end_idx] = np.array(self.next_actions_buffer)
        data_group['dones'][start_idx:end_idx] = np.array(self.dones_buffer)
        
        # 更新计数
        self.current_transitions_count += buffer_size
        self.total_transitions += buffer_size
        
        # 清空缓冲区
        self.states_buffer.clear()
        self.actions_buffer.clear()
        self.rewards_buffer.clear()
        self.next_states_buffer.clear()
        self.next_actions_buffer.clear()
        self.dones_buffer.clear()
        
        print(f"写入 {buffer_size} 个转换到文件 {self.current_file_index} "
              f"(总数: {self.current_transitions_count}/{self.transitions_per_file})")
        
    def _finalize_current_file(self):
        """完成当前文件的写入"""
        if self.current_file is None:
            return
            
        # 添加统计信息
        meta_group = self.current_file['metadata']
        meta_group.attrs['actual_transitions'] = self.current_transitions_count
        meta_group.attrs['completion_time'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 计算奖励统计
        if self.current_transitions_count > 0:
            rewards = self.current_file['transitions']['rewards'][:]
            dones = self.current_file['transitions']['dones'][:]
            meta_group.attrs['reward_mean'] = float(np.mean(rewards))
            meta_group.attrs['reward_std'] = float(np.std(rewards))
            meta_group.attrs['reward_min'] = float(np.min(rewards))
            meta_group.attrs['reward_max'] = float(np.max(rewards))
            meta_group.attrs['num_episodes'] = int(np.sum(dones))  # 统计episode数量
        
        print(f"完成文件 {self.current_file_path} (包含 {self.current_transitions_count} 个转换)")
        
    def finalize(self):
        """完成所有文件的写入"""
        # Flush剩余的缓冲区数据
        if self.states_buffer:
            self._flush_buffer()
            
        # 完成当前文件
        if self.current_file is not None:
            self._finalize_current_file()
            self.current_file.close()
            self.current_file = None
            
        # 创建总结文件
        self._create_summary_file()
        
        print(f"\n=== 数据集创建完成 ===")
        print(f"总文件数: {self.total_files}")
        print(f"总转换数: {self.total_transitions}")
        print(f"输出目录: {self.output_dir}")
        
    def _create_summary_file(self):
        """创建包含所有文件信息的总结文件"""
        summary_path = self.output_dir / "dataset_summary.yaml"
        
        summary_data = {
            'dataset_info': {
                'total_files': self.total_files,
                'total_transitions': self.total_transitions,
                'transitions_per_file': self.transitions_per_file,
                'creation_time': datetime.datetime.now().isoformat(),
                'format': 'offline_rl_transitions'
            },
            'metadata': self.metadata,
            'files': []
        }
        
        # 收集所有文件信息
        for h5_file in sorted(self.output_dir.glob("offline_rl_dataset_*.h5")):
            try:
                with h5py.File(h5_file, 'r') as f:
                    file_info = {
                        'filename': h5_file.name,
                        'file_index': int(f['metadata'].attrs['file_index']),
                        'transitions': int(f['metadata'].attrs['actual_transitions']),
                        'creation_time': str(f['metadata'].attrs['creation_time'])
                    }
                    
                    if 'reward_mean' in f['metadata'].attrs:
                        file_info['reward_stats'] = {
                            'mean': float(f['metadata'].attrs['reward_mean']),
                            'std': float(f['metadata'].attrs['reward_std']),
                            'min': float(f['metadata'].attrs['reward_min']),
                            'max': float(f['metadata'].attrs['reward_max'])
                        }
                    
                    if 'num_episodes' in f['metadata'].attrs:
                        file_info['num_episodes'] = int(f['metadata'].attrs['num_episodes'])
                    
                    summary_data['files'].append(file_info)
                    
            except Exception as e:
                logging.warning(f"Failed to read file {h5_file}: {e}")
                
        # 保存总结文件
        with open(summary_path, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, indent=2)
            
        print(f"创建总结文件: {summary_path}")
        
    def set_metadata(self, metadata: Dict):
        """设置元数据"""
        self.metadata.update(metadata)

def test_single_trajectory(device='cpu'):
    """测试处理单条轨迹"""
    processor = TrajectoryProcessor(device=device)
    
    # 查找轨迹文件
    traj_files = processor.find_trajectory_files()
    
    if not traj_files:
        print("未找到轨迹文件！")
        return
    
    print(f"找到 {len(traj_files)} 个轨迹文件")
    
    # 选择第一个文件进行测试
    test_file = traj_files[0]
    print(f"测试文件: {test_file}")
    
    try:
        # 处理轨迹
        tuples = processor.process_single_trajectory(test_file)
        
        if not tuples:
            print("未生成任何转换！")
            return
        
        # 显示示例数据
        print(f"\n=== 示例转换 ===")
        state, action, reward, next_state, next_action, done = tuples[0]
        print(f"状态形状: {state.shape}")
        print(f"动作形状: {action.shape}")
        print(f"奖励: {reward:.3f}")
        print(f"下一个状态形状: {next_state.shape}")
        print(f"下一个动作形状: {next_action.shape}")
        print(f"终止标志: {done}")
        
        print(f"\n示例状态 (前10维): {state[:10]}")
        print(f"示例动作: {action}")
        
        # 保存处理后的数据
        output_dir = Path(test_file).parent
        output_path = output_dir / f"processed_offline_rl_{Path(test_file).stem}.pkl"
        
        # 获取原始元数据
        data = processor.load_trajectory_data(test_file)
        
        processor.save_processed_data(tuples, str(output_path), data['metadata'])
        
        print(f"\n=== 测试成功完成！ ===")
        
    except Exception as e:
        print(f"处理轨迹时出错: {e}")
        import traceback
        traceback.print_exc()

def process_multiple_trajectories(device='cpu', max_files=None, world_ids=None, 
                                output_dir=None, transitions_per_file=100000):
    """批量处理多个轨迹文件，输出HDF5格式数据集"""
    processor = TrajectoryProcessor(device=device)
    
    # 设置输出目录
    if output_dir is None:
        output_dir = "/home/zkr/Documents/verti_bench/offline_rl_dataset"
    
    # 创建数据集创建器
    dataset_creator = OfflineRLDatasetCreator(output_dir, transitions_per_file)
    
    # 设置元数据
    dataset_creator.set_metadata({
        'device_used': device,
        'swae_enabled': processor.use_swae,
        'patch_size': processor.patch_size,
        'max_speed': processor.max_speed,
        'observation_dim': 18,
        'action_dim': 2
    })
    
    # 查找轨迹文件
    traj_files = processor.find_trajectory_files()
    
    if not traj_files:
        print("未找到轨迹文件！")
        return
    
    # 筛选特定世界的文件
    if world_ids:
        filtered_files = []
        for world_id in world_ids:
            world_files = [f for f in traj_files if f"/{world_id}/" in f]
            filtered_files.extend(world_files)
        traj_files = filtered_files
    
    # 限制文件数量
    if max_files is not None and max_files != -1:
        traj_files = traj_files[:max_files]
    
    print(f"找到 {len(traj_files)} 个轨迹文件")
    if max_files == -1:
        print("处理所有找到的文件")
    elif max_files is not None:
        print(f"限制处理前 {max_files} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"每个HDF5文件包含: {transitions_per_file:,} 个转换")
    
    successful_files = 0
    failed_files = 0
    
    try:
        for i, traj_file in enumerate(traj_files):
            print(f"\n处理文件 {i+1}/{len(traj_files)}: {os.path.basename(traj_file)}")
            
            try:
                transitions = processor.process_single_trajectory(traj_file)
                
                if transitions:
                    # 批量添加转换到数据集
                    dataset_creator.add_transitions_batch(transitions)
                    successful_files += 1
                    print(f"  ✓ 添加 {len(transitions)} 个转换到数据集")
                else:
                    failed_files += 1
                    print(f"  ✗ 处理失败")
                    
            except Exception as e:
                failed_files += 1
                print(f"  ✗ 处理失败: {e}")
                
        print(f"\n=== 批处理结果 ===")
        print(f"成功处理: {successful_files}/{len(traj_files)} 文件")
        print(f"失败: {failed_files} 文件")
        
    finally:
        # 完成数据集创建
        dataset_creator.finalize()
        
    return dataset_creator

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="轨迹数据处理器：转换为离线强化学习格式")
    parser.add_argument("--device", default="cpu", help="计算设备 (cpu/cuda)")
    parser.add_argument("--mode", choices=["test", "batch"], default="test", 
                       help="运行模式: test=测试单个文件, batch=批量处理")
    parser.add_argument("--max_files", type=int, default=-1,
                       help="最大处理文件数量，-1表示处理所有文件")
    parser.add_argument("--world_ids", type=int, nargs="+", 
                       help="指定处理的世界ID，例如: --world_ids 1 2 3")
    parser.add_argument("--output_dir", default="/home/zkr/Documents/verti_bench/offline_rl_dataset",
                       help="输出目录")
    parser.add_argument("--transitions_per_file", type=int, default=100000,
                       help="每个HDF5文件包含的转换数量")
    
    args = parser.parse_args()
    
    print("=== 轨迹数据处理器：转换为离线强化学习格式 ===")
    print(f"使用设备: {args.device}")
    print(f"运行模式: {args.mode}")
    
    # 检查CUDA可用性
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"✓ CUDA可用，GPU: {torch.cuda.get_device_name()}")
    elif args.device == "cuda":
        print("⚠ CUDA不可用，将使用CPU")
        args.device = "cpu"
    
    if args.mode == "test":
        # 测试单条轨迹处理
        test_single_trajectory(device=args.device)
    elif args.mode == "batch":
        # 批量处理轨迹
        print(f"批处理参数:")
        print(f"  最大文件数: {'所有文件' if args.max_files == -1 else args.max_files}")
        print(f"  世界ID: {args.world_ids if args.world_ids else '所有世界'}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  每文件转换数: {args.transitions_per_file:,}")
        
        process_multiple_trajectories(
            device=args.device, 
            max_files=args.max_files,
            world_ids=args.world_ids,
            output_dir=args.output_dir,
            transitions_per_file=args.transitions_per_file
        )

if __name__ == "__main__":
    main()
