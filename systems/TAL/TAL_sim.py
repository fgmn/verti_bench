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
import h5py
import datetime
from collections import defaultdict
from PIL import Image

from verti_bench.envs.terrain import TerrainManager
from verti_bench.systems.TAL.TAL import TALPlanner
from verti_bench.vehicles.HMMWV import HMMWVManager
from verti_bench.vehicles.FEDA import FEDAManager
from verti_bench.vehicles.Gator import GatorManager
from verti_bench.vehicles.MAN5t import MAN5tManager
from verti_bench.vehicles.MAN7t import MAN7tManager
from verti_bench.vehicles.MAN10t import MAN10tManager
from verti_bench.vehicles.M113 import M113Manager
from verti_bench.vehicles.ART import ARTManager
from verti_bench.vehicles.VW import VWManager

class TrajectoryCollector:
    """HDF5-based trajectory data collector for real-time simulation"""
    
    def __init__(self, world_id, vehicle_type, log_frequency=10.0, buffer_size=50, 
                 collect_local_terrain=True, terrain_region_size=64):
        self.world_id = world_id
        self.vehicle_type = vehicle_type
        self.log_frequency = log_frequency  # Hz
        self.log_interval = 1.0 / log_frequency
        self.buffer_size = buffer_size
        
        # Local terrain collection settings
        self.collect_local_terrain = collect_local_terrain
        self.terrain_region_size = terrain_region_size  # Size of local terrain patch
        
        # Timing control
        self.last_log_time = 0.0
        self.timestep = 0
        
        # Data buffers
        self.states_buffer = defaultdict(list)
        self.actions_buffer = defaultdict(list)
        self.terrain_buffer = defaultdict(list)  # New buffer for local terrain
        
        # File info
        self.hdf5_filename = None
        self.pos_id = None
        
    def create_trajectory_file(self, pos_id, start_pos, goal_pos, terrain_type, terrain_manager=None):
        """Create HDF5 trajectory file with metadata"""
        self.pos_id = pos_id
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Create directory
        traj_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                               "../../envs/data/BenchMaps/sampled_maps/Trajs_HDF5", str(self.world_id))
        os.makedirs(traj_dir, exist_ok=True)
        
        # Include vehicle type in filename: trajectory_world_pos_vehicle_timestamp.h5
        self.hdf5_filename = os.path.join(traj_dir, f"trajectory_{self.world_id}_{pos_id}_{self.vehicle_type}_{timestamp}.h5")
        
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
            
            # Store terrain information if available
            if terrain_manager:
                # meta_group.attrs['terrain_size'] = terrain_manager.terrain_size
                meta_group.attrs['scale_factor'] = terrain_manager.scale_factor
                
                # Store elevation map as static data
                if hasattr(terrain_manager, 'high_res_data') and terrain_manager.high_res_data is not None:
                    terrain_group = f.create_group('terrain')
                    terrain_group.create_dataset('elevation_map', data=terrain_manager.high_res_data, 
                                                compression='gzip', compression_opts=9)
                    terrain_group.attrs['elevation_resolution'] = getattr(terrain_manager, 'elevation_resolution', 1.0)
                    terrain_group.attrs['elevation_origin'] = getattr(terrain_manager, 'elevation_origin', [0.0, 0.0])
                
                # Store obstacle map if available
                if hasattr(terrain_manager, 'obs_path') and terrain_manager.obs_path:
                    try:
                        obstacle_array = np.array(Image.open(terrain_manager.obs_path))
                        terrain_group = f.get('terrain') or f.create_group('terrain')
                        terrain_group.create_dataset('obstacle_map', data=obstacle_array,
                                                    compression='gzip', compression_opts=9)
                    except Exception as e:
                        logging.warning(f"Failed to store obstacle map: {e}")
            
            # Create states group with expandable datasets
            states_group = f.create_group('states')
            max_size = None  # Unlimited size
            
            states_group.create_dataset('timestep', (0,), maxshape=(max_size,), dtype='i4')
            states_group.create_dataset('time', (0,), maxshape=(max_size,), dtype='f8')
            states_group.create_dataset('position', (0, 3), maxshape=(max_size, 3), dtype='f8')
            states_group.create_dataset('orientation', (0, 3), maxshape=(max_size, 3), dtype='f8')
            states_group.create_dataset('velocity', (0, 3), maxshape=(max_size, 3), dtype='f8')
            states_group.create_dataset('angular_velocity', (0, 3), maxshape=(max_size, 3), dtype='f8')
            states_group.create_dataset('distance_to_goal', (0,), maxshape=(max_size,), dtype='f8')
            states_group.create_dataset('local_goal', (0, 2), maxshape=(max_size, 2), dtype='f8')
            
            # Add terrain-related state data
            states_group.create_dataset('ground_height', (0,), maxshape=(max_size,), dtype='f8')
            states_group.create_dataset('terrain_normal', (0, 3), maxshape=(max_size, 3), dtype='f8')
            states_group.create_dataset('slope_angle', (0,), maxshape=(max_size,), dtype='f8')
            
            # Create local terrain group for per-frame terrain patches
            if self.collect_local_terrain:
                local_terrain_group = f.create_group('local_terrain')
                region_size = self.terrain_region_size
                
                # Local elevation map around vehicle (64x64 or configurable size)
                local_terrain_group.create_dataset('elevation_patches', 
                                                  (0, region_size, region_size), 
                                                  maxshape=(max_size, region_size, region_size), 
                                                  dtype='f4', compression='gzip', compression_opts=6)
                
                # Optional: local semantic maps if available
                local_terrain_group.create_dataset('semantic_patches', 
                                                  (0, region_size, region_size, 3), 
                                                  maxshape=(max_size, region_size, region_size, 3), 
                                                  dtype='uint8', compression='gzip', compression_opts=6)
                
                # Terrain patch metadata
                local_terrain_group.create_dataset('patch_center_world', 
                                                  (0, 2), maxshape=(max_size, 2), dtype='f8')
                local_terrain_group.create_dataset('patch_resolution', 
                                                  (0,), maxshape=(max_size,), dtype='f4')
                local_terrain_group.create_dataset('vehicle_heading', 
                                                  (0,), maxshape=(max_size,), dtype='f4')
                
                # Store region size in metadata
                meta_group.attrs['terrain_region_size'] = region_size
                meta_group.attrs['collect_local_terrain'] = True
            
            # Create actions group with expandable datasets
            actions_group = f.create_group('actions')
            actions_group.create_dataset('steering', (0,), maxshape=(max_size,), dtype='f8')
            actions_group.create_dataset('throttle', (0,), maxshape=(max_size,), dtype='f8')
            actions_group.create_dataset('braking', (0,), maxshape=(max_size,), dtype='f8')
            actions_group.create_dataset('target_speed', (0,), maxshape=(max_size,), dtype='f8')
            
        print(f"Created HDF5 trajectory file: {self.hdf5_filename}")
        return self.hdf5_filename
        
    def should_collect(self, current_time):
        """Check if it's time to collect data"""
        return (current_time - self.last_log_time) >= self.log_interval
        
    def add_data(self, state_data, action_data, terrain_data=None):
        """Add data to buffers"""
        # Add to buffers
        for key, value in state_data.items():
            self.states_buffer[key].append(value)
        for key, value in action_data.items():
            self.actions_buffer[key].append(value)
            
        # Add terrain data if available
        if terrain_data is not None and self.collect_local_terrain:
            for key, value in terrain_data.items():
                self.terrain_buffer[key].append(value)
            
        # Flush to file if buffer is full
        if len(self.states_buffer['timestep']) >= self.buffer_size:
            self.flush_to_hdf5()
            
    def flush_to_hdf5(self):
        """Write buffered data to HDF5 file"""
        if not self.states_buffer or not self.hdf5_filename:
            return
            
        try:
            with h5py.File(self.hdf5_filename, 'a') as f:
                states_group = f['states']
                actions_group = f['actions']
                
                # Get current size and calculate new size
                current_size = states_group['timestep'].shape[0]
                buffer_size = len(self.states_buffer['timestep'])
                new_size = current_size + buffer_size
                
                # Resize and write states data
                for dataset_name in states_group.keys():
                    dataset = states_group[dataset_name]
                    data_array = np.array(self.states_buffer[dataset_name])
                    
                    # Resize dataset
                    if len(data_array.shape) == 1:
                        dataset.resize((new_size,))
                    else:
                        dataset.resize((new_size,) + data_array.shape[1:])
                    
                    # Write data
                    dataset[current_size:new_size] = data_array
                
                # Resize and write actions data
                for dataset_name in actions_group.keys():
                    dataset = actions_group[dataset_name]
                    data_array = np.array(self.actions_buffer[dataset_name])
                    dataset.resize((new_size,))
                    dataset[current_size:new_size] = data_array
                
                # Resize and write local terrain data
                if self.collect_local_terrain and 'local_terrain' in f and self.terrain_buffer:
                    local_terrain_group = f['local_terrain']
                    
                    for dataset_name in local_terrain_group.keys():
                        if dataset_name in self.terrain_buffer:
                            dataset = local_terrain_group[dataset_name]
                            data_array = np.array(self.terrain_buffer[dataset_name])
                            
                            # Resize dataset based on data dimensions
                            if len(data_array.shape) == 1:
                                dataset.resize((new_size,))
                            elif len(data_array.shape) == 2:  # 2D data like patch_center_world
                                dataset.resize((new_size, data_array.shape[1]))
                            elif len(data_array.shape) == 3:  # 3D data like elevation_patches
                                dataset.resize((new_size, data_array.shape[1], data_array.shape[2]))
                            elif len(data_array.shape) == 4:  # 4D data like semantic_patches
                                dataset.resize((new_size, data_array.shape[1], data_array.shape[2], data_array.shape[3]))
                            
                            # Write data
                            dataset[current_size:new_size] = data_array
                    
            # Clear buffers
            self.states_buffer.clear()
            self.actions_buffer.clear()
            self.terrain_buffer.clear()
            
        except Exception as e:
            logging.error(f"Failed to write trajectory data to HDF5: {e}")
    
    def extract_local_terrain(self, vehicle, vehicle_pos, elevation_map=None, semantic_map=None, 
                              terrain_length=64.5, terrain_width=64.5):
        """Extract local terrain patch around vehicle position
        
        Args:
            vehicle: PyChrono vehicle object
            vehicle_pos: Vehicle position [x, y, z]
            elevation_map: Global elevation map (height data)
            semantic_map: Global semantic map (RGB data)
            
        Returns:
            dict: Terrain data containing patches and metadata
        """
        if not self.collect_local_terrain:
            return {}
            
        try:
            terrain_data = {}
            region_size = self.terrain_region_size
            
            # Get vehicle heading
            vehicle_heading = vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ().z
            terrain_data['vehicle_heading'] = float(vehicle_heading)
            terrain_data['patch_center_world'] = [float(vehicle_pos[0]), float(vehicle_pos[1])]
            
            # Extract elevation patch if elevation map is available
            if elevation_map is not None:
                elevation_patch = self._extract_terrain_patch(
                    elevation_map, vehicle_pos, vehicle_heading, region_size, 
                    terrain_length=terrain_length, terrain_width=terrain_width, is_elevation=True
                )
                terrain_data['elevation_patches'] = elevation_patch
                terrain_data['patch_resolution'] = max(terrain_length, terrain_width) / min(elevation_map.shape[:2])  # meters per pixel
            else:
                # Create empty patch if no elevation data
                terrain_data['elevation_patches'] = np.zeros((region_size, region_size), dtype=np.float32)
                terrain_data['patch_resolution'] = 1.0
            
            # Skip semantic patch collection for now
            # Create empty RGB patch (placeholder for future semantic data)
            terrain_data['semantic_patches'] = np.zeros((region_size, region_size, 3), dtype=np.uint8)
            
            return terrain_data
            
        except Exception as e:
            logging.warning(f"Failed to extract local terrain: {e}")
            # Return empty data in case of error
            return {
                'elevation_patches': np.zeros((region_size, region_size), dtype=np.float32),
                'semantic_patches': np.zeros((region_size, region_size, 3), dtype=np.uint8),
                'patch_center_world': [float(vehicle_pos[0]), float(vehicle_pos[1])],
                'patch_resolution': 1.0,
                'vehicle_heading': 0.0
            }
    
    def _extract_terrain_patch(self, terrain_map, vehicle_pos, vehicle_heading, region_size, 
                              terrain_length=64.5, terrain_width=64.5, is_elevation=True):
        """Extract and rotate terrain patch around vehicle position
        
        Args:
            terrain_map: 2D or 3D numpy array representing terrain
            vehicle_pos: Vehicle position [x, y, z] 
            vehicle_heading: Vehicle heading angle in radians
            region_size: Size of extracted patch (region_size x region_size)
            is_elevation: True for elevation data, False for semantic/RGB data
            
        Returns:
            numpy.ndarray: Extracted and rotated terrain patch
        """
        try:
            # Use provided terrain dimensions instead of hardcoded values
            # These values should match the actual terrain configuration
            map_height, map_width = terrain_map.shape[:2]
            
            # Transform vehicle position to map coordinates (following gather_trajectories logic)
            # PyChrono coordinate transformation
            vehicle_x = vehicle_pos[0]  # PyChrono X (Forward)
            vehicle_y = -vehicle_pos[1]  # PyChrono Y (Left, flipped)
            
            # Normalization factors (same as in gather_trajectories.py)
            s_norm_x = map_width / (2 * terrain_length)
            s_norm_y = map_height / (2 * terrain_width)
            # print("\033[33m" + f"s_norm_x: {s_norm_x}, s_norm_y: {s_norm_y}" + "\033[0m")
            # s_norm_x: 10.007751937984496, s_norm_y: 10.007751937984496

            # Transform to bitmap coordinates
            pos_chrono = np.array([vehicle_x + terrain_length, vehicle_y + terrain_width, 1])
            
            # Apply transformation matrix
            pixel_x = int(np.round(np.clip(pos_chrono[0] * s_norm_x, 0, map_width - 1)))
            pixel_y = int(np.round(np.clip(pos_chrono[1] * s_norm_y, 0, map_height - 1)))
            
            # Center the vehicle position in the map
            center_x = map_width // 2
            center_y = map_height // 2
            shift_x = center_x - pixel_x
            shift_y = center_y - pixel_y
            
            # Shift the map to center the vehicle position
            # if is_elevation:
            #     shifted_map = np.roll(terrain_map, shift_y, axis=0)  # y shift affects rows (axis 0)
            #     shifted_map = np.roll(shifted_map, shift_x, axis=1)  # x shift affects columns (axis 1)
            # else:
            #     shifted_map = np.roll(terrain_map, shift_y, axis=0)
            #     shifted_map = np.roll(shifted_map, shift_x, axis=1)
            
            # 使得车辆位置位于地图中心
            shifted_map = np.roll(terrain_map, shift_y, axis=0)
            shifted_map = np.roll(shifted_map, shift_x, axis=1)

            # Rotate the map based on vehicle heading (following gather_trajectories logic)
            angle_degrees = np.degrees(vehicle_heading) % 360
            
            try:
                import torch
                import torchvision.transforms.functional as TF
                
                # Using tensor to accelerate the rotation process (corrected for newer PyTorch)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                if is_elevation:
                    # 使patch始终与车辆朝向对齐
                    # 无论车辆实际朝向如何，旋转后的patch总是以车辆正前方为"上"方向
                    tensor_map = torch.tensor(shifted_map, device=device).unsqueeze(0).float()
                    rotated_tensor = TF.rotate(tensor_map, -angle_degrees)
                    rotated_map = rotated_tensor.squeeze().cpu().numpy()
                    rotated_map = np.fliplr(rotated_map)  # Apply flip as in gather_trajectories
                else:
                    tensor_map = torch.tensor(shifted_map, device=device).permute(2, 0, 1).float()
                    rotated_tensor = TF.rotate(tensor_map, -angle_degrees)
                    rotated_map = rotated_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    rotated_map = np.fliplr(rotated_map)
                
            except (ImportError, AttributeError):
                # Fallback to scipy rotation if PyTorch is not available or has issues
                try:
                    from scipy.ndimage import rotate
                    rotated_map = rotate(shifted_map, -angle_degrees, reshape=False, order=1, mode='nearest')
                    rotated_map = np.fliplr(rotated_map)
                except ImportError:
                    logging.warning("Neither PyTorch nor scipy available for terrain patch rotation, using unrotated map")
                    rotated_map = shifted_map
            
            # Extract the region around the vehicle center (following gather_trajectories logic)
            center_y, center_x = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
            half_size = region_size // 2
            
            start_y = center_y - half_size
            end_y = center_y + half_size
            start_x = center_x - half_size
            end_x = center_x + half_size
            
            # Handle boundary conditions
            start_x = max(0, start_x)
            end_x = min(rotated_map.shape[1], end_x)
            start_y = max(0, start_y)
            end_y = min(rotated_map.shape[0], end_y)
            
            # Extract patch
            if is_elevation:
                patch = rotated_map[start_y:end_y, start_x:end_x]
                patch = patch.T  # Transpose as in gather_trajectories
            else:
                patch = rotated_map[start_y:end_y, start_x:end_x]
                patch = np.transpose(patch, (1, 0, 2))  # Transpose as in gather_trajectories
            
            # Pad patch if it's smaller than required size (near boundaries)
            if patch.shape[0] < region_size or patch.shape[1] < region_size:
                if is_elevation:
                    padded_patch = np.zeros((region_size, region_size), dtype=patch.dtype)
                    # Fill with mean elevation if available
                    fill_value = np.mean(patch) if patch.size > 0 else 0.0
                    padded_patch.fill(fill_value)
                else:
                    padded_patch = np.zeros((region_size, region_size, patch.shape[2]), dtype=patch.dtype)
                
                # Calculate paste position
                paste_start_y = (region_size - patch.shape[0]) // 2
                paste_start_x = (region_size - patch.shape[1]) // 2
                
                if is_elevation:
                    padded_patch[paste_start_y:paste_start_y + patch.shape[0],
                               paste_start_x:paste_start_x + patch.shape[1]] = patch
                else:
                    padded_patch[paste_start_y:paste_start_y + patch.shape[0],
                               paste_start_x:paste_start_x + patch.shape[1], :] = patch
                
                patch = padded_patch
            
            return patch.astype(np.float32 if is_elevation else np.uint8)
                
        except Exception as e:
            logging.warning(f"Error in terrain patch extraction: {e}")
            # Return empty patch in case of error
            if is_elevation:
                return np.zeros((region_size, region_size), dtype=np.float32)
            else:
                return np.zeros((region_size, region_size, 3), dtype=np.uint8)
            
    def finalize(self):
        """Finalize trajectory file and flush remaining data"""
        if self.states_buffer:
            self.flush_to_hdf5()
            
        if self.hdf5_filename:
            try:
                with h5py.File(self.hdf5_filename, 'a') as f:
                    # Update total timesteps in metadata
                    f['metadata'].attrs['total_timesteps'] = f['states']['timestep'].shape[0]
                    f['metadata'].attrs['completion_time'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                    
                print(f"Finalized trajectory file: {self.hdf5_filename}")
            except Exception as e:
                logging.error(f"Failed to finalize trajectory file: {e}")
    
    # def _pad_to_size(self, patch, target_size):
    #     """Pad patch to target size"""
    #     current_h, current_w = patch.shape[:2]
    #     if current_h >= target_size and current_w >= target_size:
    #         return patch[:target_size, :target_size]
        
    #     # Calculate padding
    #     pad_h = max(0, target_size - current_h)
    #     pad_w = max(0, target_size - current_w)
        
    #     if len(patch.shape) == 2:
    #         padded = np.pad(patch, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), 
    #                        mode='edge')
    #     else:
    #         padded = np.pad(patch, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2), (0, 0)), 
    #                        mode='edge')
        
    #     return padded[:target_size, :target_size]
    
    # def _rotate_patch(self, patch, angle_degrees):
    #     """Rotate terrain patch by given angle"""
    #     try:
    #         from scipy.ndimage import rotate
    #         return rotate(patch, angle_degrees, reshape=False, order=1, mode='nearest')
    #     except ImportError:
    #         logging.warning("scipy not available, skipping patch rotation")
    #         return patch
    
    # def _elevation_to_semantic(self, elevation_patch):
    #     """Convert elevation data to a simple semantic visualization"""
    #     # Normalize elevation to 0-255 range
    #     elev_min = elevation_patch.min()
    #     elev_max = elevation_patch.max()
        
    #     if elev_max > elev_min:
    #         normalized = ((elevation_patch - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
    #     else:
    #         normalized = np.zeros_like(elevation_patch, dtype=np.uint8)
        
    #     # Create RGB semantic patch (simple terrain coloring)
    #     semantic = np.zeros((elevation_patch.shape[0], elevation_patch.shape[1], 3), dtype=np.uint8)
        
    #     # Low elevation = blue (water-like), high elevation = brown/green (terrain)
    #     semantic[:, :, 0] = normalized  # Red channel for height
    #     semantic[:, :, 1] = 255 - normalized  # Green channel (inverse height)
    #     semantic[:, :, 2] = normalized // 2  # Blue channel
        
    #     return semantic

class TALSim:
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
        self.vehicle_type_lower = self.vehicle_type.lower()
        
        # Initialize system
        self.system = chrono.ChSystemNSC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        
        # Set thread counts based on available CPUs
        num_procs = multiprocessing.cpu_count()
        num_threads_chrono = min(8, num_procs)
        num_threads_collision = min(8, num_procs)
        num_threads_eigen = 1
        self.system.SetNumThreads(num_threads_chrono, num_threads_collision, num_threads_eigen)
        
        # Simulation parameters
        self.step_size = self._step_size()
        self.vis_freq = 100.0
        self.vis_dur = 1.0 / self.vis_freq
        self.last_vis_time = 0.0
        self.mppi_freq = 20.0  # Hz
        self.mppi_dur = 1.0 / self.mppi_freq
        self.last_mppi_time = 0.0
        self.last_replan_time = 0.0
        self.replan_interval = 0.1
        self.vis = None
        self.driver = None
        
        # Stuck tracking
        self.stuck_counter = 0
        self.stuck_distance = 0.01
        self.stuck_time = self._stuck_time()
        self.last_position = None
        
        # Load terrain configs
        self.terrain_manager = TerrainManager(self.world_id, self.scale_factor)
            
        # Create planner
        self._initialize_system()
        
        # Create vehicle manager
        self._initialize_vehicle()
        
        # Initialize trajectory collector
        self.trajectory_collector = None
        self.collect_trajectory = config.get('collect_trajectory', False)
        if self.collect_trajectory:
            log_freq = config.get('trajectory_log_frequency', 10.0)
            buffer_size = config.get('trajectory_buffer_size', 50)
            collect_local_terrain = config.get('collect_local_terrain', True)
            terrain_region_size = config.get('terrain_region_size', 64)
            
            self.trajectory_collector = TrajectoryCollector(
                self.world_id, 
                self.vehicle_type, 
                log_freq, 
                buffer_size,
                collect_local_terrain,
                terrain_region_size
            )
    
    def _step_size(self):
        """Get vehicle-specific step size based on vehicle type"""
        step_sizes = {
            'hmmwv': 5e-3,
            'gator': 2e-3,
            'feda': 1e-3,
            'man5t': 1e-3,
            'man7t': 1e-3,
            'man10t': 1e-3,
            'm113': 8e-4,
            'art': 1e-3,
            'vw': 3e-4,
            'default': 1e-3       
        }
        
        return step_sizes[self.vehicle_type_lower]
    
    def _stuck_time(self):
        """
        Get stuck time
        """
        stuck_time = {
            'hmmwv': 10,
            'gator': 40,
            'feda': 40,
            'man5t': 50,
            'man7t': 50,
            'man10t': 60,
            'm113': 60,
            'art': 60,
            'vw': 60,
            'default': 10       
        }
        
        return stuck_time[self.vehicle_type_lower]
        
    def _initialize_system(self):
        """Initialize the control system based on system_type"""
        if self.system_type.lower() == 'tal':
            self.planner = TALPlanner(self.terrain_manager)
        else:
            raise ValueError(f"Unsupported system type: {self.system_type}.")
    
    def _initialize_vehicle(self):
        """Initialize the vehicle manager based on vehicle_type"""
        if self.vehicle_type.lower() == 'hmmwv':
            self.vehicle_manager = HMMWVManager(self.system, self.step_size)
        elif self.vehicle_type.lower() == 'gator':
            self.vehicle_manager = GatorManager(self.system, self.step_size)
        elif self.vehicle_type.lower() == 'feda':
            self.vehicle_manager = FEDAManager(self.system, self.step_size)
        elif self.vehicle_type.lower() == 'man5t':
            self.vehicle_manager = MAN5tManager(self.system, self.step_size)    
        elif self.vehicle_type.lower() == 'man7t':
            self.vehicle_manager = MAN7tManager(self.system, self.step_size)    
        elif self.vehicle_type.lower() == 'man10t':
            self.vehicle_manager = MAN10tManager(self.system, self.step_size)  
        elif self.vehicle_type.lower() == 'm113':
            self.vehicle_manager = M113Manager(self.system, self.step_size)  
        elif self.vehicle_type.lower() == 'art':
            self.vehicle_manager = ARTManager(self.system, self.step_size)  
        elif self.vehicle_type.lower() == 'vw':
            self.vehicle_manager = VWManager(self.system, self.step_size)
        else:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}. ")
        
    def _setup_visualization(self):
        """Set up visualization system"""
        self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        if self.vehicle_type_lower in ['m113']:
            self.vis =veh.ChTrackedVehicleVisualSystemIrrlicht()
        self.vis.SetWindowTitle('vws in the wild')
        self.vis.SetWindowSize(3840, 2160)
        
        if self.vehicle_type_lower in ['man5t', 'man7t', 'man10t']:
            trackPoint = chrono.ChVector3d(-5.5, 0.0, 2.6)
        elif self.vehicle_type_lower in ['hmmwv', 'gator', 'feda', 'm113']:
            trackPoint = chrono.ChVector3d(-1.0, 0.0, 1.75)
        else:
            trackPoint = chrono.ChVector3d(2.0, 0.0, 0.0)
        self.vis.SetChaseCamera(trackPoint, 6.0, 0.5)
        self.vis.Initialize()
        self.vis.AddLightDirectional()
        self.vis.AddSkyBox()
        self.vis.AttachVehicle(self.vehicle_manager.vehicle.GetVehicle())
        self.vis.EnableStats(True)
        
    def _setup_driver(self):
        """Set up driver (interactive or autonomous)"""
        if self.use_gui:  
            self.driver = veh.ChInteractiveDriverIRR(self.vis)
            self.driver.SetSteeringDelta(0.1)
            self.driver.SetThrottleDelta(0.02)
            self.driver.SetBrakingDelta(0.06)
            self.driver.Initialize()
        else:
            self.driver = veh.ChDriver(self.vehicle_manager.vehicle.GetVehicle())
            
        self.driver_inputs = self.driver.GetInputs()
            
    def _collect_state_data(self, time):
        """Collect current state data for trajectory"""
        vehicle_pos = self.vehicle_manager.get_position()
        euler_angles = self.vehicle_manager.get_rotation()
        
        # Get velocity using chassis body (following gather_trajectories.py pattern)
        chassis_body = self.vehicle_manager.get_chassis_body()
        velocity = chassis_body.GetPosDt()
        
        # Get angular velocity components (following gather_trajectories.py pattern)
        roll_rate = self.vehicle_manager.vehicle.GetVehicle().GetRollRate()
        pitch_rate = self.vehicle_manager.vehicle.GetVehicle().GetPitchRate()
        yaw_rate = self.vehicle_manager.vehicle.GetVehicle().GetYawRate()
        angular_velocity = chrono.ChVector3d(roll_rate, pitch_rate, yaw_rate)
        
        # Calculate distance to goal
        vector_to_goal = self.vehicle_manager.goal - vehicle_pos
        distance_to_goal = vector_to_goal.Length()
        
        # Get local goal if available
        local_goal = [0.0, 0.0]
        if hasattr(self, 'local_goal_idx') and hasattr(self, 'chrono_path'):
            try:
                if self.chrono_path and self.local_goal_idx < len(self.chrono_path):
                    local_goal_point = self.chrono_path[self.local_goal_idx]
                    local_goal = [local_goal_point[0], local_goal_point[1]]
            except (IndexError, AttributeError):
                pass
        
        # Get terrain information at current position
        ground_height = 0.0
        terrain_normal = np.array([0.0, 0.0, 1.0])  # Default to flat ground
        slope_angle = 0.0
        
        try:
            # Get ground height from terrain
            if hasattr(self, 'terrains') and self.terrains:
                for terrain in self.terrains:
                    if hasattr(terrain, 'GetHeight'):
                        ground_height = terrain.GetHeight(chrono.ChVector2d(vehicle_pos.x, vehicle_pos.y))
                        break
                    
            # Calculate terrain normal and slope
            if hasattr(self.terrain_manager, 'elevation_map') and self.terrain_manager.elevation_map is not None:
                # Convert world coordinates to elevation map indices
                # Use the actual terrain dimensions from terrain manager
                terrain_length = getattr(self.terrain_manager, 'terrain_length', 200.0) / self.terrain_manager.scale_factor
                terrain_width = getattr(self.terrain_manager, 'terrain_width', 200.0) / self.terrain_manager.scale_factor
                # For square terrain, use the maximum dimension
                terrain_size = max(terrain_length, terrain_width)
                map_size = self.terrain_manager.elevation_map.shape[0]
                scale = map_size / terrain_size
                
                map_x = int((vehicle_pos.x + terrain_size/2) * scale)
                map_y = int((vehicle_pos.y + terrain_size/2) * scale)
                
                if 0 <= map_x < map_size-1 and 0 <= map_y < map_size-1:
                    # Calculate gradient using finite differences
                    dx = self.terrain_manager.elevation_map[map_y, map_x+1] - self.terrain_manager.elevation_map[map_y, map_x-1] if map_x > 0 else 0
                    dy = self.terrain_manager.elevation_map[map_y+1, map_x] - self.terrain_manager.elevation_map[map_y-1, map_x] if map_y > 0 else 0
                    
                    # Calculate normal vector
                    pixel_size = terrain_size / map_size
                    normal_x = -dx / (2 * pixel_size)
                    normal_y = -dy / (2 * pixel_size)
                    normal_z = 1.0
                    
                    # Normalize
                    normal_length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
                    if normal_length > 0:
                        terrain_normal = np.array([normal_x/normal_length, normal_y/normal_length, normal_z/normal_length])
                        
                        # Calculate slope angle (angle from vertical)
                        slope_angle = np.arccos(np.clip(normal_z/normal_length, -1.0, 1.0))
                        
        except Exception as e:
            logging.debug(f"Failed to get terrain information: {e}")
        
        return {
            'timestep': self.trajectory_collector.timestep,           # 时间步数 [无量纲] - 仿真离散时间步计数器
            'time': time,                                             # 仿真时间 [s] - 从仿真开始的累计时间
            'position': np.array([vehicle_pos.x, vehicle_pos.y, vehicle_pos.z]),                    # 车辆位置 [m] - 世界坐标系下的三维位置 (x前进, y左侧, z向上)
            'orientation': np.array([euler_angles.x, euler_angles.y, euler_angles.z]),             # 车辆姿态 [rad] - 欧拉角 (roll横滚, pitch俯仰, yaw偏航)
            'velocity': np.array([velocity.x, velocity.y, velocity.z]),                            # 车辆线速度 [m/s] - 世界坐标系下的三维速度向量
            'angular_velocity': np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z]),  # 车辆角速度 [rad/s] - 绕xyz轴的旋转速度 (roll_rate, pitch_rate, yaw_rate)
            'distance_to_goal': distance_to_goal,                     # 到目标距离 [m] - 车辆当前位置到最终目标的直线距离
            'local_goal': np.array(local_goal),                       # 局部目标点 [m] - 路径规划中的当前子目标坐标 (x, y)
            'ground_height': ground_height,                           # 地面高度 [m] - 车辆下方地形的高程值
            'terrain_normal': terrain_normal,                         # 地形法向量 [无量纲] - 车辆下方地形表面的单位法向量 (nx, ny, nz)
            'slope_angle': slope_angle                                # 坡度角 [rad] - 地形表面相对于水平面的倾斜角度 (0=平地, π/2=垂直)
        }
    
    def _collect_action_data(self):
        """Collect current action data for trajectory"""
        return {
            'steering': self.driver_inputs.m_steering,          # 转向输入 [无量纲] - 归一化转向角度 (-1.0=右转, +1.0=左转)
            'throttle': self.driver_inputs.m_throttle,          # 油门输入 [无量纲] - 归一化油门开度 (0.0=无油门, 1.0=全油门)
            'braking': self.driver_inputs.m_braking,            # 制动输入 [无量纲] - 归一化制动力度 (0.0=无制动, 1.0=全制动)
            'target_speed': getattr(self, 'target_speed', 0.0)  # 目标速度 [m/s] - 控制器期望达到的车辆速度
        }
            
    def initialize(self, start_pos=None, goal_pos=None):
        """Initialize simulation"""
        # Use positions from config
        if start_pos is None or goal_pos is None:
            positions = self.terrain_manager.positions
            pos_id = random.randint(0, len(positions) - 1)
            selected_pair = positions[pos_id]
            start_pos = [i * self.scale_factor for i in selected_pair['start']]
            goal_pos = [i * self.scale_factor for i in selected_pair['goal']]
        else:
            pos_id = 0  # Default for manually specified positions
        
        # Store position ID for trajectory collection
        self.pos_id = pos_id
        
        # Initialize terrain
        self.terrains = self.terrain_manager.initialize_terrain(self.system)
        
        # Initialize vehicle
        self.vehicle_manager.initialize_vehicle(start_pos, goal_pos, self.terrain_manager)
        
        # Set up trajectory collection
        if self.collect_trajectory and self.trajectory_collector:
            self.trajectory_collector.create_trajectory_file(
                pos_id, 
                start_pos, 
                goal_pos, 
                self.terrain_manager.terrain_type,
                self.terrain_manager  # Pass terrain manager for elevation map
            )
        
        # Set up moving patches if needed
        if self.terrain_manager.terrain_type == 'deformable' or self.terrain_manager.terrain_type == 'mixed':
            if self.vehicle_type.lower() in ['m113']:
                deform_terrains = [t for t in self.terrains if isinstance(t, veh.SCMTerrain)]
                self.vehicle_manager.setup_moving_patches(deform_terrains, True)
            else:
                deform_terrains = [t for t in self.terrains if isinstance(t, veh.SCMTerrain)]
                self.vehicle_manager.setup_moving_patches(deform_terrains, False)
        
        # Create obstacle map and plan path
        obs_path = self.terrain_manager.obs_path
        obstacle_array = np.array(Image.open(obs_path))
        self.planner.set_obstacle_map(obstacle_array)
        self.chrono_path = self.planner.astar_path(obs_path, start_pos, goal_pos)
        self.local_goal_idx = 0
        
        # Set up visualization
        if self.render:
            self._setup_visualization()
            
        # Set up driver
        self._setup_driver()
            
    def run(self):
        """Run the simulation"""
        # Check initialization
        if not hasattr(self, 'terrains') or not self.terrains:
            raise ValueError("Simulation not initialized. Call initialize() first.")
            
        # Initialize timing
        start_time = self.system.GetChTime()
        roll_angles = []
        pitch_angles = []
        
        # Main simulation loop
        while True:
            time = self.system.GetChTime()
            
            # Handle visualization if enabled
            if self.render:
                if not self.vis.Run():
                    break
                    
                if self.last_vis_time == 0 or (time - self.last_vis_time) > self.vis_dur:
                    self.vis.BeginScene()
                    self.vis.Render()
                    self.vis.EndScene()
                    self.last_vis_time = time
            
            # Get vehicle position and orientation
            vehicle_pos = self.vehicle_manager.get_position()
            vector_to_goal = self.vehicle_manager.goal - vehicle_pos
            
            # Update controls
            if self.use_gui:
                self.driver_inputs = self.driver.GetInputs()
            else:
                # Get vehicle orientation
                euler_angles = self.vehicle_manager.get_rotation()
                roll = euler_angles.x
                pitch = euler_angles.y
                vehicle_heading = euler_angles.z
                roll_angles.append(np.degrees(abs(roll)))
                pitch_angles.append(np.degrees(abs(pitch)))
                
                # Check if replanning is needed
                if time - self.last_replan_time >= self.replan_interval:
                    # print("Replanning path...")
                    obs_path = self.terrain_manager.obs_path
                    new_path = self.planner.astar_replan(
                        obs_path,
                        (vehicle_pos.x, vehicle_pos.y),
                        (self.vehicle_manager.goal.x, self.vehicle_manager.goal.y)
                    )
                    
                    if new_path is not None:
                        self.chrono_path = new_path
                        self.local_goal_idx = 0
                        # print("Path replanned successfully")
                    else:
                        pass
                        # print("Failed to replan path!")
                    
                    self.last_replan_time = time
                
                if self.last_mppi_time == 0 or (time - self.last_mppi_time) >= self.mppi_dur:
                    # Find local goal along the path
                    self.local_goal_idx, local_goal = self.planner.find_local_goal(
                        (vehicle_pos.x, vehicle_pos.y), 
                        vehicle_heading,
                        self.chrono_path, 
                        self.local_goal_idx
                    )

                    # Update goal and odometry in MPPI
                    self.planner.goal_cb(local_goal)
                    self.planner.odom_cb(self.vehicle_manager, self.system)
                    
                    # Update elevation map
                    obstacle_array = np.array(Image.open(self.terrain_manager.obs_path))
                    self.planner.gridMap_callback(self.vehicle_manager.vehicle, vehicle_pos, obstacle_array)
                    
                    # Run MPPI update
                    self.planner.mppi_cb(self.planner.curr_pose, self.planner.pose_dot)
                    
                    # Get control commands
                    speed, steer = self.planner.send_controls()
                    
                    # Scale speed appropriately for the vehicle
                    speed = speed * 10.0
                    
                    self.last_mppi_time = time
                    
                # Apply steering with rate limiting
                self.driver_inputs.m_steering = np.clip(steer, -1, 1)
                
                # Compute throttle and braking
                throttle, braking = self.planner.compute_throttle(
                    self.speed, 
                    time, 
                    self.step_size,
                    self.vehicle_manager.vehicle.GetVehicle().GetRefFrame()
                )
                
                self.driver_inputs.m_throttle = throttle
                self.driver_inputs.m_braking = braking
            
            # Collect trajectory data if enabled
            if (self.collect_trajectory and self.trajectory_collector and 
                self.trajectory_collector.should_collect(time)):
                
                try:
                    state_data = self._collect_state_data(time)
                    action_data = self._collect_action_data()
                    
                    # Collect local terrain data if enabled (elevation only for now)
                    terrain_data = None
                    if self.trajectory_collector.collect_local_terrain:
                        # Get high resolution elevation data from terrain manager
                        elevation_map = getattr(self.terrain_manager, 'high_res_data', None) # 1291^2
                        # Skip semantic map for now
                        # semantic_map = getattr(self.terrain_manager, 'high_sem_data', None)
                        
                        terrain_data = self.trajectory_collector.extract_local_terrain(
                            self.vehicle_manager.vehicle,
                            (vehicle_pos.x, vehicle_pos.y, vehicle_pos.z),
                            elevation_map=elevation_map,
                            semantic_map=None,  # Not using semantic maps for now
                            terrain_length=self.terrain_manager.terrain_length / self.terrain_manager.scale_factor,
                            terrain_width=self.terrain_manager.terrain_width / self.terrain_manager.scale_factor
                        )
                    
                    self.trajectory_collector.add_data(state_data, action_data, terrain_data)
                    self.trajectory_collector.last_log_time = time
                    self.trajectory_collector.timestep += 1
                except Exception as e:
                    logging.warning(f"Failed to collect trajectory data: {e}")
            
            # Check if vehicle is stuck or reached goal
            current_position = (vehicle_pos.x, vehicle_pos.y, vehicle_pos.z)
            
            if self.last_position:
                position_change = np.sqrt(
                    (current_position[0] - self.last_position[0])**2 +
                    (current_position[1] - self.last_position[1])**2 +
                    (current_position[2] - self.last_position[2])**2
                )
                
                if position_change < self.stuck_distance:
                    self.stuck_counter += self.step_size
                else:
                    self.stuck_counter = 0
                    
                if self.stuck_counter >= self.stuck_time:
                    print('--------------------------------------------------------------')
                    print('Vehicle stuck!')
                    print(f'Stuck time: {self.stuck_counter:.2f} seconds')
                    print(f'Position change: {position_change:.3f} m')
                    print(f'Initial position: {self.vehicle_manager.init_loc}')
                    print(f'Current position: {vehicle_pos}')
                    print(f'Goal position: {self.vehicle_manager.goal}')
                    print(f'Distance to goal: {vector_to_goal.Length():.2f} m')
                    print('--------------------------------------------------------------')
                    
                    if self.render:
                        self.vis.Quit()
                        
                    # Finalize trajectory collection
                    if self.collect_trajectory and self.trajectory_collector:
                        self.trajectory_collector.finalize()
                        
                    avg_roll = np.mean(roll_angles) if roll_angles else 0
                    avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                    return time - start_time, False, avg_roll, avg_pitch
            
            self.last_position = current_position
            
            # Check if goal reached
            if vector_to_goal.Length() < 8 * self.scale_factor:
                print('--------------------------------------------------------------')
                print('Goal Reached')
                print(f'Initial position: {self.vehicle_manager.init_loc}')
                print(f'Goal position: {self.vehicle_manager.goal}')
                print('--------------------------------------------------------------')
                
                if self.render:
                    self.vis.Quit()
                    
                # Finalize trajectory collection
                if self.collect_trajectory and self.trajectory_collector:
                    self.trajectory_collector.finalize()
                    
                avg_roll = np.mean(roll_angles) if roll_angles else 0 
                avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                return time - start_time, True, avg_roll, avg_pitch
            
            # Check if time limit exceeded
            if time > self.max_time:
                print('--------------------------------------------------------------')
                print('Time out')
                print('Initial position: ', self.vehicle_manager.init_loc)
                dist = vector_to_goal.Length()
                print('Final position of vw: ', self.vehicle_manager.chassis_body.GetPos())
                print('Goal position: ', self.vehicle_manager.goal)
                print('Distance to goal: ', dist)
                print('--------------------------------------------------------------')
                
                if self.render:
                    self.vis.Quit()
                    
                # Finalize trajectory collection
                if self.collect_trajectory and self.trajectory_collector:
                    self.trajectory_collector.finalize()
                    
                avg_roll = np.mean(roll_angles) if roll_angles else 0 
                avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                return time - start_time, False, avg_roll, avg_pitch
            
            # Synchronize terrains and vehicle
            for terrain in self.terrains:
                terrain.Synchronize(time)
                
                if self.vehicle_type_lower in ['m113']:
                    self.vehicle_manager.synchronize(time, self.driver_inputs)
                else:
                    self.vehicle_manager.synchronize(time, self.driver_inputs, terrain)
                
                terrain.Advance(self.step_size)
            
            # Advance simulation components
            self.driver.Advance(self.step_size)
            self.vehicle_manager.advance(self.step_size)
            
            if self.render:
                self.vis.Synchronize(time, self.driver_inputs)
                self.vis.Advance(self.step_size)
            
            # Step the system
            self.system.DoStepDynamics(self.step_size)
        
        # Finalize trajectory collection if loop exits unexpectedly
        if self.collect_trajectory and self.trajectory_collector:
            self.trajectory_collector.finalize()
            
        return None, False, 0, 0  # Return default values if loop exits unexpectedly
    