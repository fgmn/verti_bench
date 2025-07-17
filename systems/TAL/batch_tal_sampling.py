#!/usr/bin/env python3
"""
TAL系统批量轨迹采样脚本

功能:
- 支持多个场景(world_id)的批量采样
- 支持不同车型的测试
- 多进程并行采样提高效率
- 统计成功率、翻车率等性能指标
- 自动保存详细的性能报告

配置文件:
    使用 batch_sampling_config.yaml 配置所有参数

使用方法:
    1. 编辑 batch_sampling_config.yaml 文件配置参数
    2. 运行: python batch_tal_sampling.py
    
配置文件说明:
    - num_worlds: 要测试的世界数量
    - samples_per_world: 每个世界每种车型的采样次数  
    - vehicle_types: 要测试的车型列表
    - num_workers: 并行工作进程数
    - simulation: 仿真相关参数
    - data_collection: 数据收集参数
    - output: 输出设置
    - thresholds: 性能评估阈值
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from systems.TAL.TAL_sim import TALSim
from verti_bench.envs.utils.utils import SetChronoDataDirectories


def setup_logging(log_dir: Path, enable_console_output: bool = True) -> logging.Logger:
    """设置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_tal_sampling_{timestamp}.log"
    
    # 清除已有的处理器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    handlers = [file_handler]
    
    # 控制台处理器（可选）
    if enable_console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        # 为控制台使用更简洁的格式
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    if enable_console_output:
        logger.info(f"批量采样日志保存到: {log_file}")
        logger.info(f"子进程日志保存到: {log_dir}/subprocess/")
    return logger


def run_single_simulation(args: Tuple) -> Dict:
    """运行单个仿真任务
    
    Args:
        args: (task_id, world_id, vehicle_type, sample_id, output_dir, config)
        
    Returns:
        Dict: 仿真结果统计
    """
    task_id, world_id, vehicle_type, sample_id, output_dir, sim_config = args
    
    # 检查是否需要重定向子进程输出
    redirect_output = sim_config.get('redirect_subprocess_output', True)
    
    if redirect_output:
        # 重定向子进程输出到日志文件
        log_dir = output_dir / "logs" / "subprocess"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个子进程创建独立的日志文件
        process_log_file = log_dir / f"task_{task_id}_world_{world_id}_{vehicle_type}_{sample_id}.log"
        
        # 重定向stdout和stderr到日志文件
        import contextlib
        with open(process_log_file, 'w', encoding='utf-8') as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # 设置Chrono数据目录（每个进程都需要设置）
                try:
                    SetChronoDataDirectories()
                except:
                    pass  # 忽略重复设置的错误
                
                # 配置子进程的日志系统，只输出到文件
                import logging
                subprocess_logger = logging.getLogger(f'subprocess_{task_id}')
                
                # 设置日志级别
                log_level_str = sim_config.get('subprocess_log_level', 'WARNING')
                log_level = getattr(logging, log_level_str.upper(), logging.WARNING)
                subprocess_logger.setLevel(log_level)
                
                # 清除默认的处理器
                subprocess_logger.handlers.clear()
                
                # 添加文件处理器
                file_handler = logging.FileHandler(process_log_file, mode='a')
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                subprocess_logger.addHandler(file_handler)
                
                return _run_simulation_core(task_id, world_id, vehicle_type, sample_id, output_dir, sim_config, subprocess_logger)
    else:
        # 不重定向输出，直接执行
        try:
            SetChronoDataDirectories()
        except:
            pass
            
        import logging
        subprocess_logger = logging.getLogger(f'subprocess_{task_id}')
        return _run_simulation_core(task_id, world_id, vehicle_type, sample_id, output_dir, sim_config, subprocess_logger)


def _run_simulation_core(task_id: int, world_id: int, vehicle_type: str, sample_id: int, 
                        output_dir: Path, sim_config: Dict, logger: logging.Logger) -> Dict:
    """仿真执行的核心逻辑"""
    start_time = time.time()
    
    result = {
        'task_id': task_id,
        'world_id': world_id,
        'vehicle_type': vehicle_type,
        'sample_id': sample_id,
        'success': False,
        'duration': 0.0,
        'avg_roll': 0.0,
        'avg_pitch': 0.0,
        'max_roll': 0.0,
        'max_pitch': 0.0,
        'rollover': False,
        'trajectory_file': None,
        'error': None,
        'execution_time': 0.0,
        'distance_traveled': 0.0,
        'goal_reached': False,
        'stuck_detected': False,
        # 添加详细时间分析
        'time_breakdown': {
            'config_creation': 0.0,
            'sim_creation': 0.0,
            'sim_initialization': 0.0,
            'sim_execution': 0.0,
            'data_processing': 0.0,
            'file_io': 0.0,
            'cleanup': 0.0
        }
    }
    
    sim = None  # 确保 sim 变量在 finally 块中可用
    
    try:
        # 1. 创建仿真配置
        config_start = time.time()
        config = {
            'world_id': world_id,
            'vehicle': vehicle_type,
            'system': 'TAL',
            'scale_factor': sim_config.get('scale_factor', 1.0),
            'render': False,  # 批量采样时不启用渲染
            'use_gui': False,
            'max_time': sim_config.get('max_simulation_time', 300.0),
            'speed': sim_config.get('target_speed', 4.0),
            
            # 轨迹收集设置
            'collect_trajectory': True,
            'trajectory_log_frequency': sim_config.get('log_frequency', 10.0),
            'trajectory_buffer_size': sim_config.get('buffer_size', 50),
            'collect_local_terrain': sim_config.get('collect_local_terrain', True),
            'terrain_region_size': sim_config.get('terrain_region_size', 64),
            'trajectory_output_dir': str(output_dir)
        }
        result['time_breakdown']['config_creation'] = time.time() - config_start
        
        # 2. 创建仿真对象
        creation_start = time.time()
        sim = TALSim(config)
        result['time_breakdown']['sim_creation'] = time.time() - creation_start
        
        # 3. 初始化仿真
        init_start = time.time()
        sim.initialize()
        result['time_breakdown']['sim_initialization'] = time.time() - init_start
        
        # 4. 运行仿真
        exec_start = time.time()
        duration, success, avg_roll, avg_pitch = sim.run()
        result['time_breakdown']['sim_execution'] = time.time() - exec_start
        
        result['time_breakdown']['sim_execution'] = time.time() - exec_start
        
        # 5. 处理基本结果
        data_start = time.time()
        result['success'] = success
        result['duration'] = duration
        result['avg_roll'] = avg_roll
        result['avg_pitch'] = avg_pitch
        result['goal_reached'] = success
        
        # 6. 计算更详细的统计信息（文件I/O密集）
        file_io_start = time.time()
        if sim.trajectory_collector and sim.trajectory_collector.hdf5_filename:
            result['trajectory_file'] = sim.trajectory_collector.hdf5_filename
            
            # 尝试从轨迹文件中提取更多统计信息
            try:
                import h5py
                with h5py.File(sim.trajectory_collector.hdf5_filename, 'r') as f:
                    if 'states' in f and 'position' in f['states']:
                        positions = f['states']['position'][:]
                        if len(positions) > 1:
                            # 计算行驶距离
                            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                            result['distance_traveled'] = float(np.sum(distances))
                    
                    if 'states' in f and 'orientation' in f['states']:
                        orientations = f['states']['orientation'][:]
                        if len(orientations) > 0:
                            # 计算最大横滚角和俯仰角
                            roll_angles = np.abs(orientations[:, 0])  # Roll
                            pitch_angles = np.abs(orientations[:, 1])  # Pitch
                            
                            result['max_roll'] = float(np.max(roll_angles))
                            result['max_pitch'] = float(np.max(pitch_angles))
                            
                            # 判断是否翻车（横滚角或俯仰角超过阈值）
                            rollover_threshold_deg = sim_config.get('rollover_threshold_deg', 60.0)
                            rollover_threshold = np.radians(rollover_threshold_deg)
                            result['rollover'] = bool(np.any(roll_angles > rollover_threshold) or 
                                                    np.any(pitch_angles > rollover_threshold))
            except Exception as e:
                logger.warning(f"无法提取轨迹文件详细信息: {e}")
        
        result['time_breakdown']['file_io'] = time.time() - file_io_start
        result['time_breakdown']['data_processing'] = time.time() - data_start - result['time_breakdown']['file_io']
        
        # 检测是否卡住（成功率低可能表示卡住）
        stuck_time_ratio = sim_config.get('stuck_time_ratio', 0.8)
        if not success and duration >= config['max_time'] * stuck_time_ratio:
            result['stuck_detected'] = True
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"任务 {task_id} (World {world_id}, {vehicle_type}) 执行失败: {e}")
    
    finally:
        # 7. 清理资源
        cleanup_start = time.time()
        if sim is not None:
            try:
                # 尝试清理仿真对象
                if hasattr(sim, 'cleanup'):
                    sim.cleanup()
                elif hasattr(sim, 'finalize'):
                    sim.finalize()
                # 删除仿真对象引用
                del sim
                logger.debug(f"任务 {task_id} 仿真对象已清理")
            except Exception as cleanup_error:
                logger.warning(f"任务 {task_id} 清理仿真对象时出错: {cleanup_error}")
        result['time_breakdown']['cleanup'] = time.time() - cleanup_start
    
    result['execution_time'] = time.time() - start_time
    
    # 输出详细时间分析（仅当执行时间异常时）
    if result['execution_time'] > result['duration'] * 3:  # 如果执行时间超过仿真时间3倍
        logger.warning(f"任务 {task_id} 时间异常分析:")
        logger.warning(f"  总执行时间: {result['execution_time']:.1f}s")
        logger.warning(f"  仿真内部时间: {result['duration']:.1f}s")
        for phase, duration in result['time_breakdown'].items():
            percentage = (duration / result['execution_time']) * 100
            logger.warning(f"  {phase}: {duration:.1f}s ({percentage:.1f}%)")
    
    return result


def progress_monitor(progress_queue: mp.Queue, total_tasks: int, logger: logging.Logger):
    """进度监控进程"""
    completed = 0
    successful = 0
    failed = 0
    rollover_count = 0
    
    start_time = time.time()
    last_report_time = start_time
    
    # 打印初始状态
    print(f"\n{'='*80}")
    print(f"开始执行 {total_tasks} 个仿真任务")
    print(f"{'='*80}")
    
    while completed < total_tasks:
        try:
            # 从队列获取结果
            result = progress_queue.get(timeout=5.0)
            completed += 1
            
            if result['success']:
                successful += 1
            else:
                failed += 1
            
            if result.get('rollover', False):
                rollover_count += 1
            
            # 每10个任务或每30秒报告一次进度
            current_time = time.time()
            if (completed % 10 == 0 or 
                current_time - last_report_time > 30 or 
                completed == total_tasks):
                
                progress = completed / total_tasks * 100
                elapsed_time = current_time - start_time
                
                if completed > 0:
                    avg_time_per_task = elapsed_time / completed
                    remaining_tasks = total_tasks - completed
                    eta_seconds = avg_time_per_task * remaining_tasks
                    eta_minutes = eta_seconds / 60
                    
                    success_rate = successful / completed * 100
                    rollover_rate = rollover_count / completed * 100
                    
                    # 使用清晰的进度条格式
                    progress_bar = "█" * int(progress // 2) + "░" * (50 - int(progress // 2))
                    
                    print(f"\r[{progress_bar}] {progress:6.1f}% | "
                          f"完成: {completed:4d}/{total_tasks} | "
                          f"成功率: {success_rate:5.1f}% | "
                          f"翻车率: {rollover_rate:5.1f}% | "
                          f"剩余: {eta_minutes:5.1f}分钟", end="")
                    
                    # 如果是最后一个任务或者每50个任务，换行
                    if completed == total_tasks or completed % 50 == 0:
                        print()  # 换行
                
                last_report_time = current_time
                
        except:
            # 超时继续等待
            continue
    
    # 最终统计
    final_success_rate = successful / total_tasks * 100 if total_tasks > 0 else 0
    final_rollover_rate = rollover_count / total_tasks * 100 if total_tasks > 0 else 0
    total_time_minutes = (time.time() - start_time) / 60
    
    print(f"\n{'='*80}")
    print(f"所有任务执行完成!")
    print(f"总任务数: {total_tasks} | 成功: {successful} | 失败: {failed}")
    print(f"成功率: {final_success_rate:.1f}% | 翻车率: {final_rollover_rate:.1f}%")
    print(f"总耗时: {total_time_minutes:.1f} 分钟")
    print(f"{'='*80}\n")


def analyze_results(results: List[Dict], output_dir: Path, logger: logging.Logger) -> Dict:
    """分析批量采样结果"""
    logger.info("开始分析采样结果...")
    
    # 基本统计
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r['success'])
    failed_tasks = total_tasks - successful_tasks
    rollover_tasks = sum(1 for r in results if r.get('rollover', False))
    stuck_tasks = sum(1 for r in results if r.get('stuck_detected', False))
    
    # 按车型统计
    vehicle_stats = {}
    for result in results:
        vehicle = result['vehicle_type']
        if vehicle not in vehicle_stats:
            vehicle_stats[vehicle] = {
                'total': 0, 'success': 0, 'rollover': 0, 'stuck': 0,
                'avg_duration': [], 'avg_roll': [], 'avg_pitch': [], 
                'distance_traveled': []
            }
        
        stats = vehicle_stats[vehicle]
        stats['total'] += 1
        if result['success']:
            stats['success'] += 1
        if result.get('rollover', False):
            stats['rollover'] += 1
        if result.get('stuck_detected', False):
            stats['stuck'] += 1
        
        if result['duration'] > 0:
            stats['avg_duration'].append(result['duration'])
        if result['avg_roll'] != 0:
            stats['avg_roll'].append(abs(result['avg_roll']))
        if result['avg_pitch'] != 0:
            stats['avg_pitch'].append(abs(result['avg_pitch']))
        if result.get('distance_traveled', 0) > 0:
            stats['distance_traveled'].append(result['distance_traveled'])
    
    # 计算车型平均值
    for vehicle, stats in vehicle_stats.items():
        stats['success_rate'] = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['rollover_rate'] = stats['rollover'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['stuck_rate'] = stats['stuck'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        stats['avg_duration_mean'] = np.mean(stats['avg_duration']) if stats['avg_duration'] else 0
        stats['avg_roll_mean'] = np.mean(stats['avg_roll']) if stats['avg_roll'] else 0
        stats['avg_pitch_mean'] = np.mean(stats['avg_pitch']) if stats['avg_pitch'] else 0
        stats['avg_distance_mean'] = np.mean(stats['distance_traveled']) if stats['distance_traveled'] else 0
        
        # 添加执行时间统计
        execution_times = [r['execution_time'] for r in results if r['vehicle_type'] == vehicle and r.get('execution_time', 0) > 0]
        stats['avg_execution_time'] = np.mean(execution_times) if execution_times else 0
        stats['max_execution_time'] = np.max(execution_times) if execution_times else 0
    
    # 按世界ID统计
    world_stats = {}
    for result in results:
        world_id = result['world_id']
        if world_id not in world_stats:
            world_stats[world_id] = {'total': 0, 'success': 0, 'rollover': 0}
        
        world_stats[world_id]['total'] += 1
        if result['success']:
            world_stats[world_id]['success'] += 1
        if result.get('rollover', False):
            world_stats[world_id]['rollover'] += 1
    
    # 计算世界成功率
    for world_id, stats in world_stats.items():
        stats['success_rate'] = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
        stats['rollover_rate'] = stats['rollover'] / stats['total'] * 100 if stats['total'] > 0 else 0
    
    # 汇总统计
    analysis = {
        'summary': {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'rollover_tasks': rollover_tasks,
            'stuck_tasks': stuck_tasks,
            'success_rate': successful_tasks / total_tasks * 100 if total_tasks > 0 else 0,
            'rollover_rate': rollover_tasks / total_tasks * 100 if total_tasks > 0 else 0,
            'stuck_rate': stuck_tasks / total_tasks * 100 if total_tasks > 0 else 0
        },
        'vehicle_performance': vehicle_stats,
        'world_difficulty': world_stats,
        'raw_results': results
    }
    
    # 保存分析结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_file = output_dir / f"tal_performance_analysis_{timestamp}.json"
    
    # 转换numpy类型为python原生类型以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return obj
    
    # 递归转换
    def recursive_convert(data):
        if isinstance(data, dict):
            return {k: recursive_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert_numpy(data)
    
    analysis_serializable = recursive_convert(analysis)
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"分析结果保存到: {analysis_file}")
    return analysis


def print_performance_report(analysis: Dict, logger: logging.Logger):
    """打印性能报告"""
    summary = analysis['summary']
    vehicle_perf = analysis['vehicle_performance']
    
    logger.info("="*80)
    logger.info("TAL算法性能评估报告")
    logger.info("="*80)
    
    # 总体统计
    logger.info(f"总任务数: {summary['total_tasks']}")
    logger.info(f"成功任务: {summary['successful_tasks']} ({summary['success_rate']:.1f}%)")
    logger.info(f"失败任务: {summary['failed_tasks']}")
    logger.info(f"翻车次数: {summary['rollover_tasks']} ({summary['rollover_rate']:.1f}%)")
    logger.info(f"卡住次数: {summary['stuck_tasks']} ({summary['stuck_rate']:.1f}%)")
    
    # 各车型性能
    logger.info("\n" + "-"*80)
    logger.info("各车型性能统计:")
    logger.info("-"*80)
    logger.info(f"{'车型':<10} {'成功率':<8} {'翻车率':<8} {'卡住率':<8} {'仿真时长':<10} {'实际时长':<10} {'平均距离':<10}")
    logger.info("-"*80)
    
    for vehicle, stats in vehicle_perf.items():
        logger.info(
            f"{vehicle:<10} "
            f"{stats['success_rate']:<8.1f} "
            f"{stats['rollover_rate']:<8.1f} "
            f"{stats['stuck_rate']:<8.1f} "
            f"{stats['avg_duration_mean']:<10.1f} "
            f"{stats.get('avg_execution_time', 0):<10.1f} "
            f"{stats['avg_distance_mean']:<10.1f}"
        )
    
    logger.info("="*80)
    
    # 添加时间分析
    logger.info("\n" + "="*80)
    logger.info("时间性能分析:")
    logger.info("="*80)
    
    # 计算总的执行时间统计
    all_execution_times = [r.get('execution_time', 0) for r in analysis['raw_results'] if r.get('execution_time', 0) > 0]
    all_sim_durations = [r.get('duration', 0) for r in analysis['raw_results'] if r.get('duration', 0) > 0]
    
    if all_execution_times:
        total_cpu_time = sum(all_execution_times)
        avg_task_time = np.mean(all_execution_times)
        max_task_time = np.max(all_execution_times)
        min_task_time = np.min(all_execution_times)
        
        logger.info(f"平均任务执行时间: {avg_task_time:.1f} 秒")
        logger.info(f"最长任务执行时间: {max_task_time:.1f} 秒")
        logger.info(f"最短任务执行时间: {min_task_time:.1f} 秒")
        logger.info(f"总CPU时间: {total_cpu_time/60:.1f} 分钟")
        
        # 理论并行效率分析
        theoretical_time = total_cpu_time / 4  # 4个进程
        logger.info(f"理论并行时间: {theoretical_time/60:.1f} 分钟")
    
    if all_sim_durations:
        avg_sim_time = np.mean(all_sim_durations)
        logger.info(f"平均仿真内部时间: {avg_sim_time:.1f} 秒")
        
        # 计算时间效率比
        if all_execution_times:
            avg_execution_time = np.mean(all_execution_times)
            efficiency_ratio = avg_sim_time / avg_execution_time * 100
            overhead_time = avg_execution_time - avg_sim_time
            logger.info(f"仿真效率: {efficiency_ratio:.1f}% (开销时间: {overhead_time:.1f}秒)")
    
    # 详细时间分解分析
    time_breakdown_results = [r for r in analysis['raw_results'] if 'time_breakdown' in r]
    if time_breakdown_results:
        logger.info("\n" + "-"*60)
        logger.info("详细时间分解分析 (平均值):")
        logger.info("-"*60)
        
        # 计算各阶段平均时间
        phases = ['config_creation', 'sim_creation', 'sim_initialization', 'sim_execution', 'data_processing', 'file_io', 'cleanup']
        phase_names = {
            'config_creation': '配置创建',
            'sim_creation': '仿真对象创建',
            'sim_initialization': '仿真初始化',
            'sim_execution': '仿真执行',
            'data_processing': '数据处理',
            'file_io': '文件I/O',
            'cleanup': '资源清理'
        }
        
        avg_total_time = np.mean([r['execution_time'] for r in time_breakdown_results])
        
        for phase in phases:
            phase_times = [r['time_breakdown'].get(phase, 0) for r in time_breakdown_results]
            avg_phase_time = np.mean(phase_times)
            percentage = (avg_phase_time / avg_total_time) * 100
            logger.info(f"{phase_names[phase]:<10}: {avg_phase_time:6.1f}s ({percentage:5.1f}%)")
        
        # 找出最耗时的阶段
        max_phases = {}
        for phase in phases:
            phase_times = [r['time_breakdown'].get(phase, 0) for r in time_breakdown_results]
            max_phases[phase] = max(phase_times)
        
        slowest_phase = max(max_phases, key=max_phases.get)
        logger.info(f"\n最耗时阶段: {phase_names[slowest_phase]} (最大: {max_phases[slowest_phase]:.1f}秒)")
    
    logger.info("="*80)


def generate_task_list(num_worlds: int, samples_per_world: int, vehicle_types: List[str]) -> List[Tuple]:
    """生成任务列表"""
    tasks = []
    task_id = 0
    
    for world_id in range(1, num_worlds + 1):
        for vehicle_type in vehicle_types:
            for sample_id in range(samples_per_world):
                tasks.append((task_id, world_id, vehicle_type, sample_id))
                task_id += 1
    
    return tasks


def load_config(config_path: str = None) -> Dict:
    """加载YAML配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        Dict: 配置参数字典
    """
    if config_path is None:
        # 使用默认配置文件路径
        current_dir = Path(__file__).parent
        config_path = current_dir / "batch_sampling_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必要的配置项
        required_sections = ['simulation', 'data_collection', 'output', 'thresholds']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置文件缺少必要的节: {section}")
        
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件格式错误: {e}")
    except Exception as e:
        raise ValueError(f"读取配置文件失败: {e}")


def main():
    """主函数"""
    # 加载配置文件
    try:
        config = load_config()
        print(f"成功加载配置文件: batch_sampling_config.yaml")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        print("请确保配置文件 batch_sampling_config.yaml 存在且格式正确")
        return
    
    # 从配置文件中提取参数
    num_worlds = config.get('num_worlds', 10)
    samples_per_world = config.get('samples_per_world', 3)
    num_workers = config.get('num_workers', 4)
    vehicle_types = config.get('vehicle_types', ['HMMWV', 'Gator', 'FEDA'])
    
    # 仿真参数
    sim_params = config.get('simulation', {})
    max_simulation_time = sim_params.get('max_simulation_time', 300.0)
    target_speed = sim_params.get('target_speed', 4.0)
    scale_factor = sim_params.get('scale_factor', 1.0)
    
    # 数据收集参数
    data_params = config.get('data_collection', {})
    log_frequency = data_params.get('log_frequency', 10.0)
    buffer_size = data_params.get('buffer_size', 50)
    collect_local_terrain = data_params.get('collect_local_terrain', True)
    terrain_region_size = data_params.get('terrain_region_size', 64)
    
    # 输出参数
    output_params = config.get('output', {})
    base_directory = output_params.get('base_directory', 'tal_batch_results')
    include_timestamp = output_params.get('include_timestamp', True)
    redirect_subprocess_output = output_params.get('redirect_subprocess_output', True)
    
    # 日志参数
    logging_params = config.get('logging', {})
    main_process_console = logging_params.get('main_process_console', True)
    subprocess_log_level = logging_params.get('subprocess_log_level', 'WARNING')
    detailed_progress = logging_params.get('detailed_progress', True)
    
    # 阈值参数
    thresholds = config.get('thresholds', {})
    rollover_threshold_deg = thresholds.get('rollover_angle', 60.0)
    stuck_time_ratio = thresholds.get('stuck_time_ratio', 0.8)
    
    # 设置输出目录
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"{base_directory}_{timestamp}")
    else:
        output_dir = Path(base_directory)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir / "logs", enable_console_output=main_process_console)
    
    # 设置Chrono数据目录
    SetChronoDataDirectories()
    
    logger.info("="*80)
    logger.info("TAL系统批量轨迹采样开始")
    logger.info(f"测试世界数: {num_worlds}")
    logger.info(f"每世界采样数: {samples_per_world}")
    logger.info(f"车型: {', '.join(vehicle_types)}")
    logger.info(f"并行进程数: {num_workers}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*80)
    
    # 生成任务列表
    tasks = generate_task_list(num_worlds, samples_per_world, vehicle_types)
    total_tasks = len(tasks)
    logger.info(f"生成了 {total_tasks} 个采样任务")
    
    # 仿真配置
    sim_config = {
        'scale_factor': scale_factor,
        'max_simulation_time': max_simulation_time,
        'target_speed': target_speed,
        'log_frequency': log_frequency,
        'buffer_size': buffer_size,
        'collect_local_terrain': collect_local_terrain,
        'terrain_region_size': terrain_region_size,
        'rollover_threshold_deg': rollover_threshold_deg,
        'stuck_time_ratio': stuck_time_ratio,
        'redirect_subprocess_output': redirect_subprocess_output,
        'subprocess_log_level': subprocess_log_level
    }
    
    # 准备工作参数
    work_args = []
    for task_id, world_id, vehicle_type, sample_id in tasks:
        work_args.append((task_id, world_id, vehicle_type, sample_id, output_dir, sim_config))
    
    # 创建进程池和进度监控
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    # 启动进度监控进程
    monitor_process = mp.Process(target=progress_monitor, 
                                args=(progress_queue, total_tasks, logger))
    monitor_process.daemon = True  # 设置为守护进程，主进程退出时自动收尾
    monitor_process.start()
    
    start_time = time.time()
    results = []
    
    # 手动管理进程池生命周期，而不是使用 with 语句
    executor = ProcessPoolExecutor(max_workers=num_workers)
    try:
        # 提交所有任务
        future_to_task = {
            executor.submit(run_single_simulation, arg): arg[0]
            for arg in work_args
        }
        
        # 收集结果
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result(timeout=max_simulation_time + 60)
                results.append(result)
                progress_queue.put(result)
            except Exception as e:
                logger.error(f"任务 {task_id} 执行异常: {e}")
                # 创建失败结果
                error_result = {
                    'task_id': task_id,
                    'success': False,
                    'error': str(e),
                    'execution_time': max_simulation_time + 60
                }
                results.append(error_result)
                progress_queue.put(error_result)
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，取消所有未完成子任务并关闭进程池...")
        # 取消所有未完成的任务并立即关闭进程池
        try:
            # Python 3.9+ 支持 cancel_futures 参数
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python 3.8 及以下版本不支持 cancel_futures 参数
            executor.shutdown(wait=False)
    
    finally:
        # 确保进程池完全关闭，等待所有子进程退出
        try:
            executor.shutdown(wait=True)
        except:
            pass  # 如果已经关闭过，忽略异常
        
        # 停止进度监控进程
        if monitor_process.is_alive():
            monitor_process.terminate()
            monitor_process.join(timeout=5)  # 最多等待5秒
            if monitor_process.is_alive():
                logger.warning("进度监控进程未能正常退出")
        
        logger.info("所有子进程已正确回收")
    
    total_time = time.time() - start_time
    
    # 分析结果
    analysis = analyze_results(results, output_dir, logger)
    
    # 打印性能报告
    print_performance_report(analysis, logger)
    
    logger.info(f"\n批量采样完成!")
    logger.info(f"总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"轨迹文件保存在: {output_dir}")
    logger.info(f"性能分析保存在: {output_dir}/tal_performance_analysis_*.json")


if __name__ == "__main__":
    main()
