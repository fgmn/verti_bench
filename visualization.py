import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# ———— 1. 读取数据 ————
df_tal = pd.read_csv('/home/zkr/Documents/verti_bench/pid10_multi_experiment_results.csv')
df_pid = pd.read_csv('/home/zkr/Documents/verti_bench/tal10_multi_experiment_results.csv')
df_iql = pd.read_csv('/home/zkr/Documents/verti_bench/iql_multi_experiment_results.csv')
df_td3_bc = pd.read_csv('/home/zkr/Documents/verti_bench/td3_bc_multi_experiment_results.csv')

# 确保 success 列是 bool 类型
for df in (df_tal, df_pid, df_iql, df_td3_bc):
    if df['success'].dtype == object:
        df['success'] = df['success'].map({'True': True, 'False': False})

# 读取分组标签
with open('/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final/config_labels.json', 'r') as f:
    groups = json.load(f)

# ———— 2. 准备方法、分组、指标 ————
methods = {
    'TAL': df_tal,
    'PID': df_pid,
    'IQL': df_iql,
    'TD3-BC': df_td3_bc,
}

groupings = ['difficulty', 'obstacle_density', 'terrain_type']

# 定义要展示的四个指标：列名 -> (大标题, y轴标签, y 轴范围)
metrics = {
    'success':       ('Success Rate',  'Success Rate',    (0, 1)),
    'time_to_goal':  ('Time to Goal',  'Time to Goal (s)', None),
    'avg_roll_deg':  ('Average Roll',   'Roll (deg)',      None),
    'avg_pitch_deg': ('Average Pitch',  'Pitch (deg)',     None),
}

# 关闭 Matplotlib 自带工具条
plt.rcParams['toolbar'] = 'None'

# ———— 3. 对每个指标画 1×3 的图 ————
for col, (fig_title, ylabel, ylim) in metrics.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
    fig.suptitle(fig_title, fontsize=16)
    
    for ax, grouping in zip(axes, groupings):
        label_dict = groups[grouping]
        labels = list(label_dict.keys())
        N = len(labels) + 1  # +1 用来放 overall
        
        x = np.arange(N)
        width = 0.2
        
        for i, (method_name, df) in enumerate(methods.items()):
            means = []
            stds  = []
            
            # 各子标签
            for label in labels:
                world_ids = label_dict[label]
                vals = df[df['world_id'].isin(world_ids)][col].dropna().astype(float)
                means.append( vals.mean() if len(vals)>0 else 0.0 )
                stds. append( vals.std()  if len(vals)>0 else 0.0 )
            
            # overall
            all_vals = df[col].dropna().astype(float)
            means.append( all_vals.mean() if len(all_vals)>0 else 0.0 )
            stds. append( all_vals.std()  if len(all_vals)>0 else 0.0 )
            
            offset = (i-0.5)*width

            if col == 'success':
                # Success Rate 不画误差棒
                ax.bar(x+offset, means, width, label=method_name)
            else:
                # 其他指标带误差棒
                ax.bar(
                    x+offset, means, width,
                    yerr=stds,
                    capsize=4,
                    label=method_name
                )
        
        ax.set_title(grouping.replace('_',' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels+['overall'], rotation=30, ha='right')
        ax.set_xlabel(grouping.replace('_',' ').title())
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()
        
        if ylim is not None:
            ax.set_ylim(*ylim)
    
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.show()
