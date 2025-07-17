#!/usr/bin/env python3
"""
Simple obstacle map visualization script
Read and visualize obstacle BMP files
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

def find_obstacle_map(world_id, difficulty):
    """Find obstacle map based on world_id and difficulty"""
    base_path = "/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final"
    map_path = os.path.join(base_path, f"obs{world_id}_{difficulty}.bmp")
    
    if os.path.exists(map_path):
        return map_path
    else:
        return None

def visualize_obstacle_map(bmp_path):
    """Read and visualize obstacle map"""
    if not os.path.exists(bmp_path):
        print(f"File not found: {bmp_path}")
        return
    
    try:
        # 读取BMP文件
        img = Image.open(bmp_path)
        img_array = np.array(img)
        
        # 确保是灰度图
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]  # 取第一个通道
        
        # 创建可视化
        plt.figure(figsize=(10, 8))
        
        # Display map: 255 (white) for obstacles, 0-254 for passable areas
        plt.imshow(img_array, cmap='gray', origin='lower')
        plt.colorbar(label='Gray Value (255=Obstacle, 0-254=Passable)')
        plt.title(f'Obstacle Map\n{os.path.basename(bmp_path)}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Add statistics
        total_pixels = img_array.size
        obstacle_pixels = np.sum(img_array == 255)
        passable_pixels = total_pixels - obstacle_pixels
        
        plt.text(0.02, 0.98, f'Total Pixels: {total_pixels}\nObstacle: {obstacle_pixels}\nPassable: {passable_pixels}\nObstacle Ratio: {obstacle_pixels/total_pixels:.2%}', 
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print(f"Map size: {img_array.shape}")
        print(f"Obstacle pixels: {obstacle_pixels}")
        print(f"Passable pixels: {passable_pixels}")
        print(f"Obstacle ratio: {obstacle_pixels/total_pixels:.2%}")
        
    except Exception as e:
        print(f"Failed to read or visualize: {e}")

def main():
    """Main function"""
    if len(sys.argv) == 1:
        print("Usage:")
        print("  python visualize_obstacle_maps.py <bmp_file_path>")
        print("  python visualize_obstacle_maps.py <world_id> <difficulty>")
        print("\nExamples:")
        print("  python visualize_obstacle_maps.py /path/to/obs_map.bmp")
        print("  python visualize_obstacle_maps.py 1 5")
        return
    
    if len(sys.argv) == 2:
        # Direct BMP file path
        bmp_path = sys.argv[1]
        visualize_obstacle_map(bmp_path)
        
    elif len(sys.argv) == 3:
        # Find via world_id and difficulty
        try:
            world_id = int(sys.argv[1])
            difficulty = int(sys.argv[2])
            
            bmp_path = find_obstacle_map(world_id, difficulty)
            if bmp_path:
                print(f"Found obstacle map: {bmp_path}")
                visualize_obstacle_map(bmp_path)
            else:
                print(f"No obstacle map found for world_{world_id} difficulty {difficulty}")
        except ValueError:
            print("world_id and difficulty must be integers")
    else:
        print("Wrong number of arguments")

if __name__ == "__main__":
    main()
