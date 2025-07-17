#!/usr/bin/env python3
"""
Chrono 模块测试脚本
测试 PyChrono 的核心模块是否能正常导入和工作
"""

import sys
import os
import subprocess
import importlib.util

def test_conda_env():
    """检查 conda 环境是否激活"""
    print("=" * 50)
    print("检查 Conda 环境状态")
    print("=" * 50)
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✓ 当前 Conda 环境: {conda_env}")
        return True
    else:
        print("✗ 未检测到激活的 Conda 环境")
        print("请先激活 conda 环境: conda activate <your_env_name>")
        return False

def check_python_version():
    """检查 Python 版本"""
    print("\n" + "=" * 50)
    print("检查 Python 版本")
    print("=" * 50)
    
    version = sys.version
    print(f"Python 版本: {version}")
    
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 7:
        print("✓ Python 版本符合要求")
        return True
    else:
        print("✗ Python 版本过低，建议使用 Python 3.7+")
        return False

def test_module_import(module_name, display_name=None):
    """测试模块导入"""
    if display_name is None:
        display_name = module_name
    
    try:
        # 尝试导入模块
        module = importlib.import_module(module_name)
        print(f"✓ {display_name} - 导入成功")
        
        # 尝试获取版本信息（如果有的话）
        if hasattr(module, '__version__'):
            print(f"  版本: {module.__version__}")
        elif hasattr(module, 'GetVersion'):
            try:
                version = module.GetVersion()
                print(f"  版本: {version}")
            except:
                pass
                
        return True
        
    except ImportError as e:
        print(f"✗ {display_name} - 导入失败")
        print(f"  错误: {e}")
        return False
    except Exception as e:
        print(f"? {display_name} - 导入时出现异常")
        print(f"  错误: {e}")
        return False

def test_chrono_basic_functionality():
    """测试 Chrono 基本功能"""
    print("\n" + "=" * 50)
    print("测试 Chrono 基本功能")
    print("=" * 50)
    
    try:
        import pychrono.core as chrono
        
        # 创建一个简单的物理系统
        system = chrono.ChSystemNSC()
        print("✓ 成功创建物理系统")
        
        # 创建一个简单的物体
        body = chrono.ChBodyEasyBox(1, 1, 1, 1000)
        system.Add(body)
        print("✓ 成功添加物体到系统")
        
        # 执行一步仿真
        system.DoStepDynamics(0.01)
        print("✓ 成功执行仿真步骤")
        
        return True
        
    except Exception as e:
        print(f"✗ Chrono 基本功能测试失败: {e}")
        return False

def test_irrlicht_functionality():
    """测试 Irrlicht 可视化功能"""
    print("\n" + "=" * 50)
    print("测试 Irrlicht 可视化功能")
    print("=" * 50)
    
    try:
        import pychrono.core as chrono
        import pychrono.irrlicht as chronoirr
        
        # 创建可视化系统（不实际显示窗口）
        system = chrono.ChSystemNSC()
        
        # 尝试创建 Irrlicht 应用
        application = chronoirr.ChIrrApp(system, "Test", chronoirr.dimension2du(1024, 768))
        print("✓ Irrlicht 应用创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ Irrlicht 功能测试失败: {e}")
        print("  注意: 这可能是由于没有显示设备或其他环境问题")
        return False

def main():
    """主测试函数"""
    print("PyChrono 模块测试脚本")
    print("测试 Chrono 是否正确构建和安装")
    
    # 检查环境
    env_ok = test_conda_env()
    python_ok = check_python_version()
    
    if not (env_ok and python_ok):
        print("\n环境检查失败，请先解决环境问题")
        return False
    
    # 测试模块导入
    print("\n" + "=" * 50)
    print("测试 PyChrono 模块导入")
    print("=" * 50)
    
    modules_to_test = [
        ("pychrono.core", "Core 核心模块"),
        ("pychrono.irrlicht", "Irrlicht 可视化模块"),
        ("pychrono.robot", "Robot 机器人模块"),
        ("pychrono.sensor", "Sensor 传感器模块"),
        ("pychrono.vehicle", "Vehicle 车辆模块"),
    ]
    
    success_count = 0
    total_count = len(modules_to_test)
    
    for module_name, display_name in modules_to_test:
        if test_module_import(module_name, display_name):
            success_count += 1
    
    # 测试基本功能
    if success_count > 0:
        basic_func_ok = test_chrono_basic_functionality()
        if "pychrono.irrlicht" in [m[0] for m in modules_to_test]:
            irrlicht_ok = test_irrlicht_functionality()
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    print(f"模块导入成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 所有模块导入成功！Chrono 构建正确！")
        return True
    elif success_count > 0:
        print("⚠️  部分模块导入成功，可能需要检查构建配置")
        return False
    else:
        print("❌ 所有模块导入失败，请检查 Chrono 安装")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试脚本执行出错: {e}")
        sys.exit(1)
