"""
Test script to verify VertiBench environment and dataset compatibility.
This script checks:
1. Environment creation and basic functionality
2. Dataset loading and structure
3. Observation/action space consistency
"""

import sys
import os
import numpy as np
import torch
import h5py

# Add VertiBench to path
sys.path.append('/home/zkr/Documents/verti_bench')

try:
    from rl.off_road_VertiBench_offlinerl import off_road_art
    print("✓ Successfully imported VertiBench environment")
except ImportError as e:
    print(f"✗ Failed to import VertiBench environment: {e}")
    sys.exit(1)


def test_environment():
    """Test VertiBench environment creation and basic functionality."""
    print("\n=== Testing VertiBench Environment ===")
    
    try:
        # Create environment
        env = off_road_art(
            world_id=1,
            scale_factor=1.0,
            additional_render_mode='None'
        )
        print("✓ Environment created successfully")
        
        # Check spaces
        print(f"✓ Observation space: {env.observation_space}")
        print(f"✓ Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful, observation: {obs}")
        # print(f"✓ Reset successful, observation shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(f"✓ Step successful, reward: {reward}, done: {done}")
        print(f"  Next observation: {next_obs}")
        # print(f"  Next observation shape: {next_obs.shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset(dataset_path):
    """Test dataset loading and structure."""
    print(f"\n=== Testing Dataset: {dataset_path} ===")
    
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path does not exist: {dataset_path}")
        return False
    
    # Find HDF5 files
    hdf5_files = []
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith('.h5') or file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(dataset_path, file))
    elif os.path.isfile(dataset_path) and (dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5')):
        hdf5_files = [dataset_path]
    
    if not hdf5_files:
        print(f"✗ No HDF5 files found in {dataset_path}")
        return False
    
    print(f"✓ Found {len(hdf5_files)} HDF5 files")
    
    try:
        # Load first file to check structure
        sample_file = hdf5_files[0]
        print(f"✓ Checking sample file: {sample_file}")
        
        with h5py.File(sample_file, 'r') as f:
            print("✓ File opened successfully")
            print(f"  Keys: {list(f.keys())}")
            
            # Check if data is in transitions group
            if 'transitions' in f:
                transitions = f['transitions']
                print(f"  Transitions keys: {list(transitions.keys())}")
                
                # Check required keys in transitions
                required_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
                for key in required_keys:
                    if key in transitions:
                        print(f"  ✓ {key}: shape {transitions[key].shape}")
                    else:
                        print(f"  ✗ Missing key: {key}")
                        return False
                
                # Load sample data from transitions
                states = transitions['states'][:100]  # Load first 100 samples
                actions = transitions['actions'][:100]
                rewards = transitions['rewards'][:100]
                next_states = transitions['next_states'][:100]
                dones = transitions['dones'][:100]
                
            else:
                # Check required keys directly
                required_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
                for key in required_keys:
                    if key in f:
                        print(f"  ✓ {key}: shape {f[key].shape}")
                    else:
                        print(f"  ✗ Missing key: {key}")
                        return False
                
                # Load sample data
                states = f['states'][:100]  # Load first 100 samples
                actions = f['actions'][:100]
                rewards = f['rewards'][:100]
                next_states = f['next_states'][:100]
                dones = f['dones'][:100]
            
            print(f"✓ Sample data loaded successfully")
            print(f"  States range: [{states.min():.3f}, {states.max():.3f}]")
            print(f"  Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"  Rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")
            print(f"  Done rate: {dones.mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility():
    """Test environment and dataset compatibility."""
    print("\n=== Testing Environment-Dataset Compatibility ===")
    
    # Create environment
    try:
        env = off_road_art(world_id=1, scale_factor=1.0, additional_render_mode='None')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f"✓ Environment dimensions - state: {state_dim}, action: {action_dim}")
        env.close()
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    
    # Check dataset dimensions
    dataset_path = "/home/zkr/Documents/verti_bench/offline_rl_dataset"
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset path does not exist: {dataset_path}")
        return False
    
    try:
        # Find first HDF5 file
        hdf5_files = []
        for file in os.listdir(dataset_path):
            if file.endswith('.h5') or file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(dataset_path, file))
        
        if not hdf5_files:
            print(f"✗ No HDF5 files found in {dataset_path}")
            return False
        
        with h5py.File(hdf5_files[0], 'r') as f:
            # Check if data is in transitions group
            if 'transitions' in f:
                transitions = f['transitions']
                dataset_state_dim = transitions['states'].shape[1]
                dataset_action_dim = transitions['actions'].shape[1]
            else:
                dataset_state_dim = f['states'].shape[1]
                dataset_action_dim = f['actions'].shape[1]
                
            print(f"✓ Dataset dimensions - state: {dataset_state_dim}, action: {dataset_action_dim}")
            
            # Check compatibility
            if state_dim == dataset_state_dim:
                print("✓ State dimensions match")
            else:
                print(f"✗ State dimension mismatch: env={state_dim}, dataset={dataset_state_dim}")
                return False
            
            if action_dim == dataset_action_dim:
                print("✓ Action dimensions match")
            else:
                print(f"✗ Action dimension mismatch: env={action_dim}, dataset={dataset_action_dim}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("VertiBench IQL Compatibility Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test environment
    if test_environment():
        tests_passed += 1
    
    # Test dataset
    dataset_path = "/home/zkr/Documents/verti_bench/offline_rl_dataset"
    if test_dataset(dataset_path):
        tests_passed += 1
    
    # Test compatibility
    if test_compatibility():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Ready to train IQL on VertiBench.")
        return True
    else:
        print("✗ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
