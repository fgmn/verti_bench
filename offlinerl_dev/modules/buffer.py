import os
import h5py
import numpy as np
import torch
from typing import List, Tuple

TensorBatch = List[torch.Tensor]

class VertiBenchReplayBuffer:
    """Replay buffer for VertiBench offline dataset."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_vertibench_dataset(self, dataset_path: str):
        """Load VertiBench HDF5 dataset created by OfflineRLDatasetCreator."""
        print(f"Loading VertiBench dataset from: {dataset_path}")

        # Gather all .h5 files
        hdf5_files = []
        if os.path.isdir(dataset_path):
            for fn in os.listdir(dataset_path):
                if fn.endswith(('.h5', '.hdf5')):
                    hdf5_files.append(os.path.join(dataset_path, fn))
        elif os.path.isfile(dataset_path) and dataset_path.endswith(('.h5', '.hdf5')):
            hdf5_files = [dataset_path]
        else:
            raise ValueError(f"Dataset path {dataset_path!r} is not a file or directory")

        if not hdf5_files:
            raise ValueError(f"No HDF5 files found under {dataset_path!r}")

        print(f"Found {len(hdf5_files)} HDF5 files")

        # Accumulate
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        for fp in sorted(hdf5_files):
            print(f"  > Loading file: {fp}")
            with h5py.File(fp, 'r') as f:
                grp = f['transitions']
                all_states.append    (grp['states'][:]     )
                all_actions.append   (grp['actions'][:]    )
                all_rewards.append   (grp['rewards'][:]    )
                all_next_states.append(grp['next_states'][:])
                all_dones.append     (grp['dones'][:]      )
            print(f"    → {all_states[-1].shape[0]} transitions")

        # Concatenate
        obs  = np.concatenate(all_states,      axis=0)
        acs  = np.concatenate(all_actions,     axis=0)
        rews = np.concatenate(all_rewards,     axis=0)
        nxt  = np.concatenate(all_next_states, axis=0)
        dns  = np.concatenate(all_dones,       axis=0)

        print(f"Total transitions: {obs.shape[0]}")
        print(f"  observations:   {obs.shape}")
        print(f"  actions:        {acs.shape}")
        print(f"  rewards:        {rews.shape}")
        print(f"  next_observations: {nxt.shape}")
        print(f"  terminals:      {dns.shape}")

        # Store into the torch buffers exactly as before
        n = min(obs.shape[0], self._buffer_size)
        self._states[:n]      = self._to_tensor(obs[:n])
        self._actions[:n]     = self._to_tensor(acs[:n])
        self._rewards[:n]     = self._to_tensor(rews[:n].reshape(-1,1))
        self._next_states[:n] = self._to_tensor(nxt[:n])
        self._dones[:n]       = self._to_tensor(dns[:n].reshape(-1,1))

        self._pointer = n
        self._size    = n

        print(f"Loaded {n} transitions into replay buffer")
        return {
            "observations":      obs,
            "actions":           acs,
            "rewards":           rews,
            "next_observations": nxt,
            "terminals":         dns,
        }

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]