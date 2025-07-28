# build_offline_dataset.py

import os
import glob
import pickle
import h5py
import yaml
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

class TrajectoryProcessor:
    """轨迹数据处理器，输出 (s, a, r, s’, done) 转换元组"""
    def __init__(self, device='cpu'):
        self.trajs_dir = "/home/zkr/Documents/verti_bench/rl/collect_traj"
        self.device = device  # 如果不做 tensor 处理，这里可不用 torch.device

    def find_trajectory_files(self) -> List[str]:
        pattern = os.path.join(self.trajs_dir, "**", "*.h5")
        return sorted(glob.glob(pattern, recursive=True))

    def load_trajectory_data(self, hdf5_path: str) -> Dict[str, Any]:
        """从 transitions 组里读取所有 dataset"""
        with h5py.File(hdf5_path, 'r') as f:
            grp = f['transitions']
            data = {
                'states':      grp['states'][:] ,
                'next_states': grp['next_states'][:],
                'actions':     grp['actions'][:] ,
                'rewards':     grp['reward'][:]  ,  # 注意这里是 singular 'reward'
                'dones':       grp['done'][:]    ,  # 注意这里是 singular 'done'
            }
        # print(f"[Load] 从 {hdf5_path} 读取了 {len(data['states'])} 条转换")
        if len(data['states']) == 0:
            print(f"[Delete] {hdf5_path} 没有转换，删除文件")
            os.remove(hdf5_path)
        return data

    def process_single_trajectory(
        self, 
        hdf5_path: str
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        """
        输出 [(s, a, r, s', done), ...]
        """
        d = self.load_trajectory_data(hdf5_path)
        N = len(d['rewards'])
        transitions = []
        for i in range(N):
            s    = self.create_observation_vector( d['states'][i]     )
            a    = self.create_action_vector(      d['actions'][i]    )
            r    = float(d['rewards'][i])
            s2   = self.create_observation_vector( d['next_states'][i] )
            done = bool(d['dones'][i])
            transitions.append((s, a, r, s2, done))
        return transitions

    def create_observation_vector(self, state: np.ndarray) -> np.ndarray:
        # 这里直接返回原始 state；如果有更复杂的处理，可以替换
        return state.astype(np.float32)

    def create_action_vector(self, action: np.ndarray) -> np.ndarray:
        # 直接返回 steering + throttle/braking 合并后的 2 维动作
        return action.astype(np.float32)



class OfflineRLDatasetCreator:
    """将所有 transition 写入若干 HDF5 文件 (s,a,r,s’,done)，并打印关键信息"""
    def __init__(self, output_dir: str, max_transitions_per_file: int = 100_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_per_file = max_transitions_per_file
        self.file_idx = 0
        self.cur_count = 0
        self.h5 = None
        print(f"[Init] 输出目录: {self.output_dir}, 每文件最多 {self.max_per_file} 条转换")

    def _new_file(self):
        if self.h5:
            self.h5.close()
            print(f"[File Closed] 完成文件 #{self.file_idx-1}, 共写入 {self.cur_count} 条转换")
        fn = self.output_dir / f"offline_rl_{self.file_idx:03d}.h5"
        self.h5 = h5py.File(fn, 'w')
        grp = self.h5.create_group('transitions')
        grp.create_dataset('states',      (0, 18), maxshape=(self.max_per_file, 18), dtype='f4', chunks=True)
        grp.create_dataset('actions',     (0,  2), maxshape=(self.max_per_file,  2), dtype='f4', chunks=True)
        grp.create_dataset('rewards',     (0,   ), maxshape=(self.max_per_file,   ), dtype='f4', chunks=True)
        grp.create_dataset('next_states', (0, 18), maxshape=(self.max_per_file, 18), dtype='f4', chunks=True)
        grp.create_dataset('dones',       (0,   ), maxshape=(self.max_per_file,   ), dtype='bool', chunks=True)
        self.cur_count = 0
        print(f"[New File] 创建文件 #{self.file_idx}: {fn}")
        self.file_idx += 1

    def add_transitions(self, transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]):
        if self.h5 is None or self.cur_count + len(transitions) > self.max_per_file:
            self._new_file()
        grp = self.h5['transitions']
        n = len(transitions)
        old_count = self.cur_count
        new_total = old_count + n

        # resize all datasets
        for name, ds in grp.items():
            ds.resize((new_total,) + ds.shape[1:])

        # write data
        for i, (s, a, r, s2, done) in enumerate(transitions):
            idx = old_count + i
            grp['states'][idx]      = s
            grp['actions'][idx]     = a
            grp['rewards'][idx]     = r
            grp['next_states'][idx] = s2
            grp['dones'][idx]       = done

        self.cur_count = new_total
        print(f"[Write] 文件 #{self.file_idx-1} 写入 {n} 条转换 (当前共 {self.cur_count}/{self.max_per_file})")

    def finalize(self):
        if self.h5:
            self.h5.close()
            print(f"[Finalize] 最后关闭文件 #{self.file_idx-1}, 共写入 {self.cur_count} 条转换")
        print(f"=== 数据集创建完毕: 共创建 {self.file_idx} 个文件 ===")

if __name__ == "__main__":
    processor = TrajectoryProcessor(device='cpu')
    files = processor.find_trajectory_files()
    creator = OfflineRLDatasetCreator("/home/zkr/Documents/verti_bench/offline_rl_dataset")
    for p in files:
        trans = processor.process_single_trajectory(p)
        creator.add_transitions(trans)
    creator.finalize()
