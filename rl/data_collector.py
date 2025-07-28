import os
import sys
import glob
import multiprocessing
import random
import numpy as np
import logging
import yaml
import argparse
import h5py
import datetime
from collections import defaultdict
from PIL import Image

class TrajectoryCollector:
    """HDF5-based trajectory data collector for real-time simulation"""
    
    def __init__(self, world_id, vehicle_type, system, log_frequency=10.0, buffer_size=1e5):
        self.world_id = world_id
        self.vehicle_type = vehicle_type
        self.system = system
        self.log_frequency = log_frequency  # Hz
        self.log_interval = 1.0 / log_frequency
        self.buffer_size = buffer_size
        
        # Timing control
        self.last_log_time = 0.0
        self.timestep = 0
        
        # Data buffers
        self.transitions_buffer = defaultdict(list)
        
        # File info
        self.hdf5_filename = None
        self.pos_id = None
        
    def create_trajectory_file(self, pos_id, start_pos, goal_pos, terrain_type, elevation_map=None):
        """Create HDF5 trajectory file with metadata"""
        self.pos_id = pos_id
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Create directory
        traj_dir = os.path.join("/home/zkr/Documents/verti_bench/rl/collect_traj", str(self.world_id))
        os.makedirs(traj_dir, exist_ok=True)
        
        # Include vehicle type in filename: trajectory_world_pos_vehicle_timestamp.h5
        self.hdf5_filename = os.path.join(traj_dir, f"traj_{self.system}_{pos_id}_{self.vehicle_type}_{timestamp}.h5")
        
        # Initialize HDF5 file with metadata and empty datasets
        with h5py.File(self.hdf5_filename, 'w') as f:
            # Create metadata group
            meta_group = f.create_group('metadata')
            meta_group.attrs['world_id'] = self.world_id
            meta_group.attrs['pos_id'] = pos_id
            meta_group.attrs['vehicle_type'] = self.vehicle_type
            meta_group.attrs['terrain_type'] = terrain_type
            meta_group.attrs['start_pos'] = start_pos
            meta_group.attrs['goal_pos'] = goal_pos
            meta_group.attrs['log_frequency'] = self.log_frequency
            meta_group.attrs['creation_time'] = timestamp

            meta_group.create_dataset('elevation_map', data=elevation_map, compression='gzip', compression_opts=9)
            
            max_size = None  # Unlimited size
            transitions_group = f.create_group('transitions')

            # support for other states data
            transitions_group.create_dataset('states', (0, 18), maxshape=(max_size, 18), dtype='f8')
            transitions_group.create_dataset('next_states', (0, 18), maxshape=(max_size, 18), dtype='f8')
            transitions_group.create_dataset('actions', (0, 2), maxshape=(max_size, 2), dtype='f8')
            transitions_group.create_dataset('reward', (0,), maxshape=(max_size,), dtype='f8')
            transitions_group.create_dataset('done', (0,), maxshape=(max_size,), dtype='i8')
            
        print(f"Created HDF5 trajectory file: {self.hdf5_filename}")
        return self.hdf5_filename
        
    def should_collect(self, current_time):
        """Check if it's time to collect data"""
        return (current_time - self.last_log_time) >= self.log_interval
        
    def add_data(self, state_data, action_data=None, reward=None, done=None):
        """Add data to buffers"""
        for key, value in state_data.items():
            self.transitions_buffer[key].append(value)
        if action_data is not None:
            for key, value in action_data.items():
                self.transitions_buffer[key].append(value)
        if reward is not None:
            self.transitions_buffer['reward'].append(reward)
        if done is not None:
            self.transitions_buffer['done'].append(done)
            
        # # Flush to file if buffer is full
        # if len(self.states_buffer['timestep']) >= self.buffer_size:
        #     self.flush_to_hdf5()
            
    def flush_to_hdf5(self):
        """Write buffered transitions to HDF5 file and clear the buffer."""
        if not self.transitions_buffer or self.hdf5_filename is None:
            return

        try:
            with h5py.File(self.hdf5_filename, 'a') as f:
                grp = f['transitions']
                # assume at least 'states' exists to infer sizes
                current_size = grp['states'].shape[0]
                # number of new entries to append
                n = len(self.transitions_buffer['states'])
                new_size = current_size + n

                # resize all datasets and write
                for name, data_list in self.transitions_buffer.items():
                    ds = grp[name]
                    arr = np.stack(data_list, axis=0) if isinstance(data_list[0], (list, np.ndarray)) and ds.ndim > 1 \
                          else np.array(data_list)
                    # resize: first dim grows by n
                    ds.resize((new_size,) + ds.shape[1:])
                    ds[current_size:new_size] = arr

                # clear buffer
                self.transitions_buffer.clear()

                # update last_log_time so we don't flush again until enough time passes
                self.last_log_time = f.attrs.get('last_flush_time', 0.0)
                f.attrs['last_flush_time'] = self.last_log_time

        except Exception as e:
            logging.error(f"Failed to write trajectory data to HDF5: {e}")

    def finalize(self):
        """Finalize trajectory file: flush remaining data and update metadata."""
        # flush any remaining transitions
        self.flush_to_hdf5()

        if self.hdf5_filename is None:
            return

        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            with h5py.File(self.hdf5_filename, 'a') as f:
                # count how many transitions we ended up with
                total = f['transitions']['states'].shape[0]
                meta = f['metadata']
                meta.attrs['total_transitions'] = total
                meta.attrs['completion_time'] = timestamp
            print(f"Finalized trajectory file: {self.hdf5_filename} (total transitions: {total})")
        except Exception as e:
            logging.error(f"Failed to finalize trajectory file: {e}")
    