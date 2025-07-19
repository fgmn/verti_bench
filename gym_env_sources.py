#!/usr/bin/env python3
"""
Gym环境来源分析
详细说明当前系统中各种环境的来源和提供者
"""

import gym
import d4rl

def analyze_gym_sources():
    """分析Gym环境的来源"""
    
    all_envs = list(gym.envs.registry.env_specs.keys())
    
    # 按来源分类环境
    sources = {
        '🏛️ OpenAI Gym 核心': {
            'description': 'OpenAI Gym官方提供的基础环境',
            'package': 'gym',
            'envs': []
        },
        '🤖 MuJoCo官方': {
            'description': 'MuJoCo物理引擎官方环境（需要MuJoCo许可证）',
            'package': 'gym[mujoco]',
            'envs': []
        },
        '🔫 PyBullet': {
            'description': 'PyBullet物理引擎提供的环境（开源）',
            'package': 'pybullet',
            'envs': []
        },
        '📊 D4RL Bullet': {
            'description': 'D4RL项目基于PyBullet的离线RL环境',
            'package': 'd4rl',
            'envs': []
        },
        '🐜 D4RL AntMaze': {
            'description': 'D4RL项目的蚂蚁迷宫导航任务',
            'package': 'd4rl',
            'envs': []
        },
        '🗺️ D4RL Maze2D': {
            'description': 'D4RL项目的2D迷宫环境',
            'package': 'd4rl',
            'envs': []
        },
        '🍳 D4RL Kitchen': {
            'description': 'D4RL项目的机械手厨房任务',
            'package': 'd4rl',
            'envs': []
        },
        '🎮 Box2D': {
            'description': 'Box2D物理引擎环境',
            'package': 'gym[box2d]',
            'envs': []
        }
    }
    
    # 分类所有环境
    for env_id in all_envs:
        env_lower = env_id.lower()
        
        if env_id.startswith('bullet-'):
            sources['📊 D4RL Bullet']['envs'].append(env_id)
        elif 'antmaze' in env_lower:
            sources['🐜 D4RL AntMaze']['envs'].append(env_id)
        elif 'maze2d' in env_lower:
            sources['🗺️ D4RL Maze2D']['envs'].append(env_id)
        elif 'kitchen' in env_lower:
            sources['🍳 D4RL Kitchen']['envs'].append(env_id)
        elif 'bullet' in env_lower and not env_id.startswith('bullet-'):
            sources['🔫 PyBullet']['envs'].append(env_id)
        elif any(x in env_lower for x in ['bipedal', 'lunar', 'carracing']):
            sources['🎮 Box2D']['envs'].append(env_id)
        elif any(x in env_lower for x in ['halfcheetah', 'hopper', 'walker2d', 'ant', 'humanoid', 'reacher', 'pusher', 'swimmer']) and 'bullet' not in env_lower:
            # 检查是否是MuJoCo环境
            if any(x in env_id for x in ['-v2', '-v3']):
                sources['🤖 MuJoCo官方']['envs'].append(env_id)
            else:
                sources['🏛️ OpenAI Gym 核心']['envs'].append(env_id)
        else:
            sources['🏛️ OpenAI Gym 核心']['envs'].append(env_id)
    
    return sources

def get_package_info():
    """获取相关包的信息"""
    packages = {}
    
    try:
        import gym
        packages['gym'] = gym.__version__
    except ImportError:
        packages['gym'] = 'Not installed'
    
    try:
        import d4rl
        packages['d4rl'] = d4rl.__version__ if hasattr(d4rl, '__version__') else 'Unknown'
    except ImportError:
        packages['d4rl'] = 'Not installed'
    
    try:
        import pybullet
        packages['pybullet'] = pybullet.__version__ if hasattr(pybullet, '__version__') else f'API {pybullet.getAPIVersion()}'
    except ImportError:
        packages['pybullet'] = 'Not installed'
    
    try:
        import mujoco
        packages['mujoco'] = mujoco.__version__
    except ImportError:
        packages['mujoco'] = 'Not installed'
    
    return packages

def main():
    print("=" * 80)
    print("🎯 Gym环境来源分析报告")
    print("=" * 80)
    
    # 获取包信息
    packages = get_package_info()
    print("\n📦 已安装的相关包:")
    for pkg, version in packages.items():
        print(f"  • {pkg}: {version}")
    
    # 分析环境来源
    sources = analyze_gym_sources()
    
    print(f"\n🌍 环境总数: {sum(len(source['envs']) for source in sources.values())}")
    
    for source_name, source_info in sources.items():
        if source_info['envs']:
            print(f"\n{source_name}")
            print(f"  📝 描述: {source_info['description']}")
            print(f"  📦 包名: {source_info['package']}")
            print(f"  🎯 环境数: {len(source_info['envs'])}")
            
            # 完整列举所有环境
            print("  📋 完整环境列表:")
            for env in sorted(source_info['envs']):
                print(f"    • {env}")
    
    print("\n" + "=" * 80)
    print("📚 详细说明:")
    print("=" * 80)
    
    explanations = {
        "OpenAI Gym 核心": """
        • 来源: OpenAI官方维护的基础环境集合
        • 特点: 经典的RL基准测试环境
        • 包含: CartPole, MountainCar, Acrobot, Pendulum等
        • 安装: pip install gym
        """,
        
        "MuJoCo官方": """
        • 来源: MuJoCo物理引擎官方环境
        • 特点: 高精度连续控制任务
        • 包含: HalfCheetah, Hopper, Walker2d, Ant等
        • 注意: 原本需要商业许可证，现在免费但仍需注册
        """,
        
        "PyBullet": """
        • 来源: PyBullet开源物理引擎
        • 特点: 完全免费，性能良好
        • 包含: 与MuJoCo类似的环境，但使用PyBullet实现
        • 安装: pip install pybullet
        """,
        
        "D4RL项目": """
        • 来源: UC Berkeley的离线强化学习基准
        • 特点: 预收集的专家/随机/混合数据集
        • 目的: 标准化离线RL算法评估
        • 论文: https://arxiv.org/abs/2004.07219
        • 包含:
          - Bullet版本: 使用PyBullet的连续控制
          - AntMaze: 蚂蚁在迷宫中的导航
          - Maze2D: 简化的2D迷宫导航
          - Kitchen: 机械手操作厨房用具
        """,
        
        "Box2D": """
        • 来源: Box2D 2D物理引擎
        • 特点: 2D环境，计算效率高
        • 包含: BipedalWalker, LunarLander, CarRacing
        • 安装: pip install gym[box2d]
        """
    }
    
    for title, explanation in explanations.items():
        print(f"\n🔍 {title}:")
        print(explanation.strip())

if __name__ == "__main__":
    main()
