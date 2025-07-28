#!/usr/bin/env python3
"""
Test script for local terrain collection functionality in TAL simulation

This script demonstrates how to use the enhanced trajectory collection system
with local terrain patch extraction for each frame.
"""

import os
import sys
import yaml
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from verti_bench.envs.utils.utils import SetChronoDataDirectories

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from systems.TAL.TAL_sim import TALSim

def create_test_config():
    """Create test configuration with local terrain collection enabled"""
    config = {
        'world_id': 75,  # Use world 75 for testing
        'scale_factor': 1.0,
        'render': False,  # Set to True if you want to see visualization
        'use_gui': False,
        'vehicle': 'hmmwv',  # Test vehicle
        'system': 'tal',
        'max_time': 60.0,  # Short test run
        'speed': 4.0,
        
        # Trajectory collection settings
        'collect_trajectory': True,
        'trajectory_log_frequency': 5.0,  # 5 Hz for testing
        'trajectory_buffer_size': 1000,  # Small buffer for testing
        
        # Local terrain collection settings
        'collect_local_terrain': True,
        'terrain_region_size': 64,  # 64x64 terrain patches
    }
    
    return config

def analyze_trajectory_file(hdf5_path):
    """Analyze the generated trajectory file"""
    print(f"\n=== Analyzing trajectory file: {hdf5_path} ===")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Print file structure
        print("\nFile structure:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj).__name__}")
            if isinstance(obj, h5py.Dataset):
                print(f"    Shape: {obj.shape}, Dtype: {obj.dtype}")
                if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                    print(f"    Attributes: {dict(obj.attrs)}")
        
        f.visititems(print_structure)
        
        # Check metadata
        if 'metadata' in f:
            print("\nMetadata:")
            for key, value in f['metadata'].attrs.items():
                print(f"  {key}: {value}")
        
        # Check data sizes
        if 'states' in f:
            num_timesteps = f['states']['timestep'].shape[0]
            print(f"\nNumber of timesteps collected: {num_timesteps}")
            
            if num_timesteps > 0:
                print(f"Time range: {f['states']['time'][0]:.2f} - {f['states']['time'][-1]:.2f} seconds")
                
                # Sample vehicle positions
                positions = f['states']['position'][:]
                print(f"Start position: [{positions[0,0]:.2f}, {positions[0,1]:.2f}, {positions[0,2]:.2f}]")
                if num_timesteps > 1:
                    print(f"End position: [{positions[-1,0]:.2f}, {positions[-1,1]:.2f}, {positions[-1,2]:.2f}]")
        
        # Check local terrain data
        if 'local_terrain' in f:
            print("\nLocal terrain data:")
            terrain_group = f['local_terrain']
            
            for dataset_name in terrain_group.keys():
                dataset = terrain_group[dataset_name]
                print(f"  {dataset_name}: Shape {dataset.shape}, Dtype {dataset.dtype}")
                
                # Show sample data for terrain patches
                if dataset_name == 'elevation_patches' and dataset.shape[0] > 0:
                    sample_patch = dataset[0]
                    print(f"    Sample elevation patch stats:")
                    print(f"      Min: {np.min(sample_patch):.3f}, Max: {np.max(sample_patch):.3f}")
                    print(f"      Mean: {np.mean(sample_patch):.3f}, Std: {np.std(sample_patch):.3f}")
                
                # Skip semantic patch analysis for now
                # elif dataset_name == 'semantic_patches' and dataset.shape[0] > 0:
                #     sample_patch = dataset[0]
                #     print(f"    Sample semantic patch stats:")
                #     print(f"      Shape: {sample_patch.shape}")
                #     print(f"      RGB ranges: R[{np.min(sample_patch[:,:,0])}-{np.max(sample_patch[:,:,0])}], "
                #           f"G[{np.min(sample_patch[:,:,1])}-{np.max(sample_patch[:,:,1])}], "
                #           f"B[{np.min(sample_patch[:,:,2])}-{np.max(sample_patch[:,:,2])}]")

def visualize_terrain_patches(hdf5_path, num_samples=4):
    """Visualize some terrain patches from the trajectory"""
    print(f"\n=== Visualizing terrain patches ===")
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'local_terrain' not in f:
            print("No local terrain data found!")
            return
        
        terrain_group = f['local_terrain']
        
        if 'elevation_patches' not in terrain_group:
            print("No elevation patches found!")
            return
        
        elevation_patches = terrain_group['elevation_patches']
        # Skip semantic patches for now
        # semantic_patches = terrain_group.get('semantic_patches', None)
        
        num_patches = min(num_samples, elevation_patches.shape[0])
        
        if num_patches == 0:
            print("No patches to visualize!")
            return
        
        # Create visualization (only elevation patches)
        fig, axes = plt.subplots(1, num_patches, figsize=(4*num_patches, 4))
        
        if num_patches == 1:
            axes = np.array([axes])
        
        for i in range(num_patches):
            # Sample indices evenly across the trajectory
            idx = i * (elevation_patches.shape[0] - 1) // (num_patches - 1) if num_patches > 1 else 0
            
            # Plot elevation patch
            elevation_patch = elevation_patches[idx]
            im1 = axes[i].imshow(elevation_patch, cmap='terrain', origin='lower')
            axes[i].set_title(f'Elevation Patch {idx}')
            axes[i].set_xlabel('X (pixels)')
            axes[i].set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=axes[i], label='Height (m)')
            
            # Skip semantic patch plotting for now
            # if semantic_patches is not None:
            #     semantic_patch = semantic_patches[idx]
            #     axes[1, i].imshow(semantic_patch, origin='lower')
            #     axes[1, i].set_title(f'Semantic Patch {idx}')
            #     axes[1, i].set_xlabel('X (pixels)')
            #     axes[1, i].set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path(hdf5_path).parent
        vis_path = output_dir / f"terrain_patches_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"Saved terrain patches visualization to: {vis_path}")
        
        # Also show the plot
        plt.show()

def main():
    # Load configuration file
    SetChronoDataDirectories()
    
    """Run the test"""
    print("=== Testing Local Terrain Collection in TAL Simulation ===")
    
    # Create test configuration
    config = create_test_config()
    print(f"Test configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Initialize simulation
        print("\nInitializing simulation...")
        sim = TALSim(config)
        
        # Initialize with random start/goal positions
        print("Setting up simulation environment...")
        sim.initialize()
        
        # Run simulation
        print(f"Running simulation for up to {config['max_time']} seconds...")
        print("Collecting trajectory data with local terrain patches...")
        
        duration, success, avg_roll, avg_pitch = sim.run()
        
        # Print results
        print(f"\n=== Simulation Results ===")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Success: {success}")
        print(f"Average roll: {avg_roll:.2f} degrees")
        print(f"Average pitch: {avg_pitch:.2f} degrees")
        
        # Analyze trajectory file if it was created
        if sim.trajectory_collector and sim.trajectory_collector.hdf5_filename:
            hdf5_path = sim.trajectory_collector.hdf5_filename
            
            if os.path.exists(hdf5_path):
                analyze_trajectory_file(hdf5_path)
                visualize_terrain_patches(hdf5_path)
                
                print(f"\nTrajectory file saved to: {hdf5_path}")
                print("You can use this file for offline RL training!")
            else:
                print("Trajectory file was not created!")
        else:
            print("Trajectory collection was not enabled!")
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
