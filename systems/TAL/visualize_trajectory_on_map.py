#!/usr/bin/env python3
"""
可视化HDF5轨迹文件：在全局地图（elevation_map或obstacle_map）上叠加车辆轨迹
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

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

def plot_trajectory_on_map(h5_path, show=True, save_path=None):
    with h5py.File(h5_path, 'r') as f:
        # 读取全局地图
        if 'terrain' in f:
            terrain_group = f['terrain']
            if 'elevation_map' in terrain_group:
                base_map = terrain_group['elevation_map'][:]
                cmap = 'terrain'
                map_type = 'elevation'
            elif 'obstacle_map' in terrain_group:
                base_map = terrain_group['obstacle_map'][:]
                cmap = 'gray'
                map_type = 'obstacle'
            else:
                raise RuntimeError("No elevation_map or obstacle_map found in terrain group!")
        else:
            raise RuntimeError("No terrain group found in HDF5 file!")

        # 读取轨迹点
        positions = f['states']['position'][:]
        # 只取x, y
        xs = positions[:, 0]
        ys = positions[:, 1]

        # 读取地图元数据
        attrs = f['metadata'].attrs
        scale_factor = attrs.get('scale_factor', 1.0)
        terrain_length = base_map.shape[1] / (2 * scale_factor) if map_type == 'elevation' else base_map.shape[1] / 2
        terrain_width = base_map.shape[0] / (2 * scale_factor) if map_type == 'elevation' else base_map.shape[0] / 2

        # 坐标变换：将轨迹点(x, y)映射到像素坐标
        def world_to_pixel(x, y):
            # PyChrono: x正前，y左，地图y轴向下
            # elevation_map/obstacle_map: shape (H, W)
            # 轨迹点单位为米，地图中心为(terrain_length, terrain_width)
            px = int(np.round((x + terrain_length) * (base_map.shape[1] / (2 * terrain_length))))
            py = int(np.round((y + terrain_width) * (base_map.shape[0] / (2 * terrain_width))))
            # clip到地图范围
            px = np.clip(px, 0, base_map.shape[1] - 1)
            py = np.clip(py, 0, base_map.shape[0] - 1)
            return px, py

        # 轨迹点像素坐标
        pixels = np.array([world_to_pixel(x, y) for x, y in zip(xs, ys)])
        pxs, pys = pixels[:, 0], pixels[:, 1]

        # 检测A*路径key
        astar_keys = ['astar_path', 'chrono_path', 'astar_points']
        astar_pixels = None
        astar_label = None
        for key in astar_keys:
            if key in f['states']:
                astar_path = f['states'][key][:]
                if astar_path.shape[-1] >= 2:
                    astar_pixels = np.array([world_to_pixel(x, y) for x, y in astar_path[:, :2]])
                    astar_label = key
                    break
        # 绘图，风格对齐 terrain patches 可视化
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(base_map, cmap=cmap, origin='lower')
        ax.plot(pxs, pys, 'r-', linewidth=2, label='Trajectory')
        ax.scatter(pxs[0], pys[0], c='green', s=60, label='Start')
        ax.scatter(pxs[-1], pys[-1], c='blue', s=60, label='End')
        # 叠加A*路径
        if astar_pixels is not None:
            ax.plot(astar_pixels[:, 0], astar_pixels[:, 1], 'b--', linewidth=2, label=f"A* Path ({astar_label})")
        ax.set_title('Trajectory on Global Map')
        ax.set_xlabel('Map X (pixel)')
        ax.set_ylabel('Map Y (pixel)')
        ax.legend()
        if map_type == 'elevation':
            plt.colorbar(im, ax=ax, label='Height (m)')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        if show:
            plt.show()

def print_hdf5_structure(h5_path):
    """递归打印HDF5文件的所有group、dataset、shape和dtype"""
    def print_group(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DATASET] {name} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GROUP]   {name}")
    with h5py.File(h5_path, 'r') as f:
        print(f"\n=== HDF5 文件结构: {h5_path} ===")
        f.visititems(print_group)
        print("\n[metadata attrs]:")
        if 'metadata' in f:
            for k, v in f['metadata'].attrs.items():
                print(f"  {k}: {v}")
        print("\n[Done]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化HDF5轨迹文件")
    parser.add_argument("--h5_path", help="输入HDF5轨迹文件路径", default="/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Trajs_HDF5/94/trajectory_94_0_HMMWV_20250717102113.h5")
    parser.add_argument("--save", help="保存图片路径", default=None)
    parser.add_argument("--print_structure", action="store_true", help="打印HDF5文件结构")
    args = parser.parse_args()
    if args.print_structure:
        print_hdf5_structure(args.h5_path)
    else:
        plot_trajectory_on_map(args.h5_path, show=True, save_path=args.save)
        visualize_terrain_patches(args.h5_path, num_samples=4)
        analyze_trajectory_file(args.h5_path)
        print("\n=== Analysis complete ===")