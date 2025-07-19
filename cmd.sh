# python setup.py \
#   vehicle=hmmwv        \  # 车型：hmmwv,gator,feda,man5t,man7t,man10t,m113,art,vw
#   system=pid           \  # 控制/算法：pid,eh,mppi,rl,mcl,acl,wmvct,mppi6,tal,tnt
#   speed=4.0            \  # 车辆目标速度 (m/s)
#   world_id=1           \  # 1–100，对应 100 张真实地形高程图
#   scale_factor=1.0     \  # 地形尺寸缩放：1、1/6、1/10
#   max_time=60          \  # 单次仿真最大秒数
#   num_experiments=1    \  # 从当前 world 的 10 组起终点里取多少条
#   render=true          \  # true=保存 PNG/视频；false=仅数据
#   use_gui=false           # true=开 Irrlicht 窗口；false=无头

python setup.py \
  --vehicle hmmwv \
  --system tal \
  --speed 4.0 \
  --world_id 99 \
  --scale_factor 1 \
  --max_time 60 \
  --num_experiments 1 \
  --render true \
  --use_gui false

# python setup.py \
#   --vehicle hmmwv \
#   --system pid \
#   --speed 4.0 \
#   --world_id 26 \
#   --scale_factor 1 \
#   --max_time 600 \
#   --num_experiments 1 \
#   --render true \
#   --use_gui true
