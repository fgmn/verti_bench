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
from PIL import Image
import shutil
import heapq
import pickle
import time

from verti_bench.envs.terrain import TerrainManager
from verti_bench.vehicles.HMMWV import HMMWVManager
from verti_bench.rl.off_road_VertiBench_offlinerl import off_road_art
from stable_baselines3 import PPO

import torch
import cv2
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from collections import defaultdict
from scipy.ndimage import binary_dilation
from scipy.special import comb

from verti_bench.systems.TAL.models import TAL, ElevMapEncDec, PatchDecoder
from verti_bench.systems.TAL.utilities import utils
from verti_bench.systems.TAL.Grid import MapProcessor
from verti_bench.systems.TAL.traversabilityEstimator import TravEstimator
from verti_bench.envs.utils.utils import SetChronoDataDirectories

from data_collector import TrajectoryCollector

# MAX_VEL = 0.6
# MIN_VEL = 0.45
MAX_VEL = 1.0
MIN_VEL = 0.0
wm_vct = False

class TALGym:
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
        
        supported_vehicles = ['hmmwv']
        if self.vehicle_type.lower() not in supported_vehicles:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}, "
                             f"only support {supported_vehicles[0]} for RL-based systems!")

        # RL specific attributes
        self.env = None
        self.model = None
        self.obs = None
        self.total_steps = 0
        
        # Clean up tmp terrain directory
        # terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                         "../../envs/data/BenchMaps/sampled_maps/Configs/tmp")
        terrain_dir = "/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/tmp"
        if os.path.exists(terrain_dir):
            shutil.rmtree(terrain_dir)

        # ======= TAL_sim.py =========
        # Simulation parameters
        self.mppi_freq = 20.0  # Hz
        self.mppi_dur = 1.0 / self.mppi_freq
        self.last_mppi_time = 0.0
        
        # ======= TAL.py =========
        #Robot Limits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.max_del = 1
        self.min_del = -1
        self.robot_length = 0.54
        self.sampled_trajectories = []
        self.look_ahead_distance = 10.0 * self.scale_factor
        self.inflation_radius = int(4.0 * self.scale_factor)
        self.speed_controller = None 
        
        #Set module-level references
        main_module = sys.modules.get('__main__', None)
        if main_module:
            if not hasattr(main_module, 'TAL'):
                main_module.TAL = TAL
            if not hasattr(main_module, 'ElevMapEncDec'):
                main_module.ElevMapEncDec = ElevMapEncDec
            if not hasattr(main_module, 'PatchDecoder'):
                main_module.PatchDecoder = PatchDecoder
        
        #External class definations 
        self.mp = MapProcessor()                                                                    # Only for crawler
        self.util = utils() 
        
        #Utility class
        self.goal_tensor = None                                                                      # Goal tensor
        self.lasttime = None                                                                         # Last time
        # model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TAL_13_14_14.torch") # Default Model path   
        model_path = "/home/zkr/Documents/verti_bench/systems/TAL/TAL_13_14_14.torch"                                        
        # enc_dec_path = "rock_LN_TH_03-07-20-08.torch"                                               
        
        elev_map_encoder = ElevMapEncDec()
        elev_map_encoder = elev_map_encoder.cuda()
        elev_map_decoder = PatchDecoder(elev_map_encoder.map_encode)
        elev_map_decoder = elev_map_decoder.cuda()

        self.model = TAL(elev_map_encoder.map_encode, elev_map_decoder)
        self.model = self.model.cuda()
        
        state_dict = torch.load(model_path, weights_only=False).state_dict()
        self.model.load_state_dict(state_dict)                                                      # Motion model
        self.model.eval()                                                                           # Model ideally runs faster in eval mode
        self.dtype = torch.float32                                                                  # Data type
        # print("Loading:", model_path)                                                               
        # print("Model:\n",self.model)
        # print("Torch Datatype:", self.dtype)

        #------MPPI variables and constants----
        #Parameters
        self.T = 20                      # Length of rollout horizon
        self.K = 800                     # Number of sample rollouts
        self.dt = 1
        self._lambda = 0.1               # Temperature
        self.sigma = torch.Tensor([0.2, 0.5]).type(torch.float32).expand(self.T, self.K, 2)  # (T, K, 2)
        self.inv_sigma = 1 / self.sigma[0, 0, :]  # (2, )
        
        # stats_pickle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stats_rock.pickle')
        stats_pickle_path = '/home/zkr/Documents/verti_bench/systems/TAL/stats_rock.pickle'
        with open(stats_pickle_path, 'rb') as f:
            scale = pickle.load(f)
        
        self.scale_state = scale['pose_dot']
        self.offset_scale = scale['map_offset']

        self.robot_pose = None                                                                    # Robot pose
        self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype)                               # (T,K,2)
        self.poses = torch.Tensor(self.K, self.T, 6).type(self.dtype)                               # (K,T,6)
        self.fpose = torch.Tensor(self.K, 6).type(self.dtype)                                    # (K,6)
        self.last_pose = None
        self.at_goal = True
        self.curr_pose = None
        self.pose_dot = torch.zeros(6).type(self.dtype)
        self.last_t = 0
        self.map_origin = torch.Tensor([64.5, 64.5]).type(self.dtype)
        self.map_resolution = 1
        self.current_trajectories = None
        self.obstacle_map = None

        #Cost variables
        self.running_cost = torch.zeros(self.K).type(self.dtype)                                    # (K, )
        self.pose_cost = torch.Tensor(self.K).type(self.dtype)                                      # (K, )
        self.bounds_check = torch.Tensor(self.K).type(self.dtype)                                   # (K, )
        self.height_check = torch.Tensor(self.K).type(self.dtype)                                   # (K, )#ony for crawler
        self.ctrl_cost = torch.Tensor(self.K, 2).type(self.dtype)                                   # (K,2)
        self.ctrl_change = torch.Tensor(self.T,2).type(self.dtype)                                  # (T,2)
        self.euclidian_distance = torch.Tensor(self.K).type(self.dtype)                             # (K, )
        self.dist_to_goal = torch.Tensor(self.K, 6).type(self.dtype)                                # (K, )
        
        self.map_embedding = None
        self.recent_controls = np.zeros((3,2))
        self.control_i = 0
        self.msgid = 0
        self.speed_tal = 0
        self.steering_angle = 0
        self.prev_ctrl = None
        self.ctrl = torch.zeros((self.T, 2))  # Initial speed = 5.0 m/s
        self.rate_ctrl = 0
        self.cont_ctrl = True
        self.chrono_path = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_idx = [1,    3 ,     5,      7,      8,      9,      10,    11,        12,         13]
        # Weights for the 8 traversability maps
        self.weights = torch.tensor([0.15, 0.15, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.3, 0.3], dtype=self.dtype).cuda().unsqueeze(-1).unsqueeze(-1)
        # Initialize the traversability model
        self.model.to(self.device)
        self.traversability_model = TravEstimator(output_dim=(320,260)).to(self.device)
        # trav_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best_traversability_model.pth')
        trav_model_path = '/home/zkr/Documents/verti_bench/systems/TAL/best_traversability_model.pth'
        self.traversability_model.load_state_dict(torch.load(trav_model_path, map_location=self.device))
        self.traversability_model.eval()  # Set to evaluation mode
        self.traversability_model.requires_grad_(False)  # Disable gradient computation
        self.traversability_mask = None
        self.trav_map_combined = None
        self.traversability_model(torch.zeros(1, 1, 160, 130).to(self.device))

        # Initialize trajectory collector
        self.trajectory_collector = None
        self.collect_trajectory = config.get('collect_trajectory', False)
        if self.collect_trajectory:
            self.trajectory_collector = TrajectoryCollector(
                world_id=self.world_id,
                vehicle_type=self.vehicle_type,
                system=self.system_type,
                log_frequency=10.0,  # Hz
            )

    def initialize(self):
        """Initialize the RL sim"""
        self.env = off_road_art(world_id=self.world_id, scale_factor=self.scale_factor)
        self.env.m_max_time = self.max_time
        self.env.max_speed = self.speed

        # Reset the environment
        self.obs, self.info = self.env.reset()

        # Create obstacle map
        obs_path = self.env.obs_path
        obstacle_array = np.array(Image.open(obs_path))
        self.set_obstacle_map(obstacle_array)
        
        # Initialize visualization if rendering is enabled
        if self.render:
            self.env.render('follow')
        
        # Initialize tracking variables
        self.total_steps = 0

        # Set up trajectory collection
        if self.collect_trajectory and self.trajectory_collector:
            self.trajectory_collector.create_trajectory_file(
                pos_id=self.env.pos_id, 
                start_pos=self.env.start_pos, 
                goal_pos=self.env.goal_pos,
                terrain_type= self.env.terrain_type,
                elevation_map=self.env.high_res_data
            )

    def set_obstacle_map(self, obstacle_array):
        """
        Set and process the obstacle map from the main function.
        """
        self.obstacle_map = torch.tensor(obstacle_array, dtype=self.dtype).to(self.device)

    def goal_cb(self, local_goal):
        """
        Callback to set the goal directly using Chrono's goal representation.
        """
        # Convert Chrono goal to PyTorch tensor
        goal_tensor_new = torch.Tensor([
            local_goal[0],  # x
            local_goal[1],  # y
            1.6,  # z
            0,            # roll (default, not provided by Chrono goal)
            0,            # pitch (default, not provided by Chrono goal)
            0             # yaw (default, not provided by Chrono goal)
        ]).type(self.dtype)

        # Set the goal tensor and update status
        self.goal_tensor = goal_tensor_new
        self.at_goal = False

    def odom_cb(self, pos, euler_angles, timenow):
        """
        Callback to update the robot's pose using Chrono's simulation data.
        """

        roll = euler_angles.x
        pitch = euler_angles.y
        yaw = euler_angles.z

        # Update current pose first
        self.robot_pose = torch.Tensor([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw    
        ]).type(self.dtype)
        
        self.curr_pose = torch.Tensor([
                    pos.x, pos.y, pos.z,
                    roll, pitch, yaw
                ]).type(self.dtype)

        # Then handle initialization
        if self.last_pose is None:
            self.last_pose = torch.Tensor([
                pos.x, pos.y, pos.z,
                roll, pitch, yaw
            ]).type(self.dtype)
            self.lasttime = timenow
            return

        difference_from_goal = np.sqrt(((self.curr_pose.cpu())[0] - (self.goal_tensor.cpu())[0])**2 + ((self.curr_pose.cpu())[1] - (self.goal_tensor.cpu())[1])**2)
        # Adjust velocity limits based on distance to goal
        if difference_from_goal < 2:
            self.min_vel = -1  # Adjusted limits
            self.max_vel = 1
        else:
            self.min_vel = MIN_VEL
            self.max_vel = MAX_VEL

        # Update pose_dot if enough time has elapsed
        t_diff = timenow - self.last_t
        if t_diff >= 0.1:
            self.pose_dot = (self.curr_pose - self.last_pose)
            self.pose_dot[5] = self.util.clamp_angle(self.pose_dot[5])  # Clamp yaw difference
            self.last_pose = self.curr_pose
            self.last_t = timenow

    def gridMap_callback(self, vehicle, vehicle_pos, obstacle_array):
        """
        Callback to process the elevation map and update the map embedding.
        :param bmp_elevation_map: Ground truth elevation map as a [129, 129] BMP array
        """
        if self.robot_pose is not None:

            # Crop [29, 29] region around the vehicle
            crop_size = 29
            cropped_map, _ = self.env.get_cropped_map(vehicle, (vehicle_pos.x, vehicle_pos.y, vehicle_pos.z), crop_size, 5)

            input_size = (360, 360)
            resized_map = cv2.resize(cropped_map, input_size, interpolation=cv2.INTER_LANCZOS4)

            # Dynamically rescale elevation values to [0, 1.6]
            map_min = obstacle_array.min()
            map_max = obstacle_array.max()
            resized_map = (resized_map - map_min) / (map_max - map_min) * 1.6  # Normalize to [0, 1.6]
            
            # Convert to tensor
            map_d = torch.tensor(resized_map, dtype=self.dtype).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

            with torch.no_grad():
                # Generate traversability maps
                map_t = F.center_crop(map_d, (320, 260))
                map_t = F.gaussian_blur(map_t, kernel_size=3, sigma=0.2)
                map_t = F.resize(map_t, (160, 130))
                traversability_maps = self.traversability_model(map_t).squeeze()
                self.map_embedding = self.model.process_map(map_d).repeat(self.K, 1, 1, 1).cuda()
                
                # Combine traversability maps
                traversability_maps = traversability_maps[self.m_idx]
                traversability_maps = traversability_maps * self.weights
                combined_traversability_map = torch.sum(traversability_maps, dim=0).squeeze()

                elevmap_n = F.center_crop(map_d, (320, 260)).squeeze()
                combined_traversability_map[abs(elevmap_n - 0.8) > 0.425] = combined_traversability_map.max()*10
                map_image = cv2.rotate(combined_traversability_map.cpu().numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
                # cv2.imshow("Traversability Map", map_image)
                
                # cv2.waitKey(1)

                self.trav_map_combined = combined_traversability_map

        else:
            print("Warning: robot_pose is not set. Cannot process elevation map.")

    def cost(self, pose, goal, ctrl, noise, t):

        self.fpose.copy_(pose)
        self.dist_to_goal.copy_(self.fpose).sub_(goal)
        
        self.dist_to_goal[:,3] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,3])
        self.dist_to_goal[:,4] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,4])
        self.dist_to_goal[:,5] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,5])

        xy_to_goal = self.dist_to_goal[:,:2]
        self.euclidian_distance = torch.norm(xy_to_goal, p=2, dim=1)
        euclidian_distance_squared = self.euclidian_distance.pow(2)

        self.ctrl_cost.copy_(ctrl).mul_(self._lambda).mul_(self.inv_sigma).mul_(noise).mul_(0.5)
        running_cost_temp = self.ctrl_cost.abs_().sum(dim=1)
        self.running_cost.copy_(running_cost_temp)
        
        eu_min = euclidian_distance_squared.min()
        eu_max = euclidian_distance_squared.max()
        euclidian_distance_squared = (euclidian_distance_squared - eu_min) / (eu_max + 1e-6 - eu_min)

        r = self.fpose[:,3].abs_()
        r[:] -= r[:].min()
        r[:] /= r[:].max()
        r[:] = r[:] 
        p = self.fpose[:,4].abs_()
        p[:] -= p[:].min()
        p[:] /= p[:].max()
        y = self.dist_to_goal[:,5].abs_()
        
        r[r<0.17] = 0.0
        p[p<0.17] = 0.0
        
        map_o_height, map_o_width = self.obstacle_map.size()

        # Use vehicle's actual position for traversability assessment
        p_x = (map_o_height // 2 - pose[:,1]).clone().detach().to(dtype=torch.long, device=self.device)
        p_y = (map_o_width // 2 + pose[:,0]).clone().detach().to(dtype=torch.long, device=self.device)
        
        p_x[p_x<=0] = 0
        p_x[p_x>=map_o_width] = map_o_width - 1
        p_y[p_y<=0] = 0
        p_y[p_y>=map_o_height] = map_o_height -1

        obstacle_penalty = (self.obstacle_map[p_y, p_x]==255).float().to(self.running_cost.device)

        self.running_cost.add_(euclidian_distance_squared*5).add_(r*2).add_(p).add_(obstacle_penalty*4)

    def mppi(self, init_pose, init_inp):
        # init_pose (6, ) [x, y, z, r, p, y]
        
        # init_input (17,):
        #   0    1      2     3     4     5       6          7          8           9         10        11       12    13    14   15     16     
        # xdot, ydot, zdot, rdot, pdot, ywdot, sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw), vel, delta, dt, map_1, map_2
        
        t0 = time.time()
        dt = self.dt

        self.running_cost.zero_()                                                  # Zero running cost
        pose = init_pose.repeat(self.K, 1).cuda()                                  # Repeat the init pose to sample size 
        nn_input = init_inp.repeat(self.K, 1).cuda()                               # Repeat the init input to sample size
        
        # Initialize storage for this rollout's trajectories
        current_rollout = torch.zeros(self.K, self.T, 6).cuda()

        state = nn_input[:, :6]                                                    # Get the state from the input
        cmd_vel = nn_input[:, 6:8]                                                 # Get the control from the input
        map_offset = torch.zeros(self.K, 4).cuda()                                 # Get the map offset from the input
        map_offset[:, :2] = nn_input[:, 8:10]                                      # Get the map offset from the input
        map_offset[:, 2] = torch.sin(pose[:, 5])
        map_offset[:, 3] = torch.cos(pose[:, 5])

        elev_map = self.map_embedding                  # Repeat the map embedding to sample size
        torch.normal(0, self.sigma, out=self.noise)                                # Generate noise based on the sigma
        if not wm_vct:
            state[:,[0,1,2]] = state[:,[0,1,2]] * 0.1
            state = self.util.scale_in(state, self.scale_state, 0)
            map_offset = self.util.scale_in(map_offset, self.offset_scale, 1)

        # Loop the forward calculation till the horizon
        for t in range(self.T):
            cmd_vel = (self.ctrl[t] + self.noise[t]).cuda() 
            # noise_scale = max(0.3, 1.0 - t/self.T)                                 # Reduce noise over horizon
            # self.noise[t] *= noise_scale                                           # Add noise to previous control input
            cmd_vel[:, 0].clamp_(self.min_vel, self.max_vel)                       # Clamp control velocity
            cmd_vel[:, 1].clamp_(self.min_del, self.max_del)                       # Clamp control steering

            # Model query for next pose caalculation
            with torch.no_grad():
                # 切换“使用 TAL 网络”还是“用 Ackermann 模型”
                if not wm_vct:
                    out = self.model.predict(state, cmd_vel, map_offset, elev_map)
                else:
                    out = self.util.ackermann_model(cmd_vel)
            state = out
            model_output = out.detach().clone()
            
            # Scale the output to add it in pose
            if not wm_vct:
                se2_pose = pose[:, [0,1,5]].clone()
                model_out_scaled = self.util.scale_out(model_output.clone(), self.scale_state, 0)
                pose_temp, state = self.util.get_next_batch_se2(model_output, se2_pose, self.scale_state)
                
                pose[:, [0,1]] = pose_temp[:,[0,1]] 
                pose[:, 2] += model_out_scaled[:, 2]
                pose[:, 3:5] = model_out_scaled[:, 3:5] 
                map_offset = self.util.get_next_offsets_se2(map_offset, se2_pose, pose_temp, self.offset_scale)

            else:
                se2_pose = pose[:, [0,1,5]].clone()
                pose_temp, state = self.util.get_next_batch_se2(model_output, se2_pose, self.scale_state)
                pose[:, [0,1,5]] = pose_temp

            # Add to self poses
            self.poses[:,t,:] = pose.clone()
            current_rollout[:, t, :] = pose.clone()

            self.sampled_trajectories = current_rollout.cpu().numpy()

            # Calculate the cost for each pose
            self.cost(pose, self.goal_tensor, self.ctrl[t], self.noise[t], t)

        # MPPI weighing
        self.running_cost -= torch.min(self.running_cost)
        self.running_cost /= -self._lambda
        torch.exp(self.running_cost, out=self.running_cost)
        weights = self.running_cost / torch.sum(self.running_cost)+1e-6

        weights = weights.unsqueeze(1).expand(self.T, self.K, 2)
        weights_temp = weights.mul(self.noise)
        self.ctrl_change.copy_(weights_temp.sum(dim=1))
        self.ctrl += self.ctrl_change
        self.ctrl[:,0].clamp_(self.min_vel, self.max_vel)
        self.ctrl[:,1].clamp_(self.min_del, self.max_del)
        
        return self.poses

    def mppi_cb(self, curr_pose, pose_dot):
        if curr_pose is None or self.goal_tensor is None:
            return
        
        # if self.chrono_path is not None:  # assuming chrono_path is accessible
        #     self.update_path(self.chrono_path)

        roll, pitch, yaw = (self.curr_pose.cpu())[3], (self.curr_pose.cpu())[4], (self.curr_pose.cpu())[5]
        pose_dot[:3] = torch.clamp(pose_dot[:3], -self.scale_state[0], self.scale_state[0])
        pose_dot[5] = torch.clamp(pose_dot[5], -self.scale_state[2], self.scale_state[2])

        nn_input = torch.Tensor([pose_dot[0], pose_dot[1], pose_dot[2], roll, pitch, pose_dot[5],
                                0.0, 0.0, 0.1, 0.0, 0.0]).type(self.dtype)

        poses = self.mppi(curr_pose, nn_input)

        run_ctrl = None
        if not self.cont_ctrl:
            run_ctrl = self.get_control().cpu().numpy()
            self.recent_controls[self.control_i] = run_ctrl
            self.control_i = (self.control_i + 1) % self.recent_controls.shape[0]
            pub_control = self.recent_controls.mean(0)
            self.speed_tal = pub_control[0]
            self.steering_angle = pub_control[1]

    def get_control(self):
        # Apply the first control values, and shift your control trajectory
        run_ctrl = self.ctrl[0].clone()

        # shift all controls forward by 1, with last control replicated
        self.ctrl = torch.roll(self.ctrl, shifts=-1, dims=0)
        return run_ctrl

    def send_controls(self):
        """
        Sends control commands to the Chrono vehicle.
        :param vehicle: The Chrono HMMWV_Reduced vehicle object
        :param delta_steer: The steering angle from MPPI
        """
        if not self.at_goal:
            if self.cont_ctrl:  # Check if continuous control is enabled
                run_ctrl = self.get_control()
                if self.prev_ctrl is None:
                    self.prev_ctrl = run_ctrl

                # Update speed and steering
                speed = (run_ctrl[0])  # Throttle value
                steer = -float(run_ctrl[1])  # Ensure steer is a float
                self.prev_ctrl = (speed, steer)
            else:
                speed = (self.speed_tal)
                steer = -float(self.steering_angle)  # Ensure delta_steer is a float
        else:
            speed = 0.0
            steer = 0.0

        return speed, steer

    def run(self):
        """Run the simulation"""
        if self.env is None or self.model is None:
            raise ValueError("Simulation not initialized. Call initialize() first.")
        
        self.total_steps = self.env.m_max_time / self.env.m_step_size
        
        self.action = np.zeros(2, dtype=np.float32)  # Initialize action
        for step in range(int(self.total_steps)):
            get_action = False
            time = self.env.m_system.GetChTime()
            if self.last_mppi_time == 0 or (time - self.last_mppi_time) >= self.mppi_dur:
                # Get vehicle state
                # In fact, we can use the env’s internal variables and methods
                vehicle_states = self.info.get('vehicle_states', None)
                assert vehicle_states is not None, "Vehicle state not found in info"
                assert len(vehicle_states) > 0, "No vehicle states available"
                vehicle_state = vehicle_states[-1]
                pos = self.info.get('pos', None)
                euler_angles = self.info.get('euler_angles', None)
                assert pos is not None, "Vehicle position not found in info"
                assert euler_angles is not None, "Vehicle euler angles not found in info"

                # Get local goal
                local_goal = self.info.get('local_goal', None)
                local_goal = (local_goal.x, local_goal.y) if isinstance(local_goal, chrono.ChVector3d) else local_goal

                # Update goal and odometry in MPPI
                self.goal_cb(local_goal)
                self.odom_cb(pos, euler_angles, time)
                
                # Update elevation map
                obstacle_array = np.array(Image.open(self.env.obs_path))
                self.gridMap_callback(self.env.m_vehicle, self.env.m_vehicle_pos, obstacle_array)
                
                # Run MPPI update
                self.mppi_cb(self.curr_pose, self.pose_dot)
                
                # Get control commands
                speed, steer = self.send_controls()
                
                # Scale speed appropriately for the vehicle
                # speed = speed*10
                
                self.last_mppi_time = time

                action = np.array([steer, speed], dtype=np.float32)
                get_action = True

            self.action = action if get_action else self.last_action
            self.last_action = self.action

            print(f"Step {step + 1}")
            print("Action: ", action)
            self.last_obs = self.obs
            self.obs, reward, done, self.info = self.env.step(self.action)
            print("obs=", self.obs, "reward=", reward, "done=", done)
            if self.render:
                self.env.render('follow')

            # Collect trajectory data
            if self.collect_trajectory and self.trajectory_collector and \
                self.trajectory_collector.should_collect(time):
                self.trajectory_collector.add_data(
                    state_data={"states": self.obs, 
                                "next_states": self.last_obs},
                    action_data={"actions": action},
                    reward=reward,
                    done=done
                )
                self.trajectory_collector.last_log_time = time
                self.trajectory_collector.timestep += 1

            if done:
                print("Simulation ended at step:", step + 1)
                if self.collect_trajectory and self.trajectory_collector:
                    self.trajectory_collector.flush_to_hdf5()
                break
            
        # return info.get('time_to_goal'), info.get('success', False), info.get('roll_angles', []), info.get('pitch_angles', [])
        results = {
            'time_to_goal': self.info.get('time_to_goal', None),
            'success': self.info.get('success', False),
            'roll_angles': self.info.get('roll_angles', []),
            'pitch_angles': self.info.get('pitch_angles', []),
        }
        return results['time_to_goal'], results['success'], results['roll_angles'], results['pitch_angles']
    
# def single_experiment(config):
#     """Run a single experiment with the given configuration"""
#     pid_gym = TALGym(config)
#     pid_gym.initialize()
    
#     time_to_goal, success, roll_angles, pitch_angles = pid_gym.run()
    
#     # Collect results
#     result = {
#         'time_to_goal': time_to_goal,
#         'success': success,
#         'roll_angles': roll_angles,
#         'pitch_angles': pitch_angles,
#         'avg_roll': np.mean(roll_angles) if roll_angles else None,
#         'avg_pitch': np.mean(pitch_angles) if pitch_angles else None
#     }
    
#     return result

# if __name__ == "__main__":
#     # Load configuration file
#     SetChronoDataDirectories()

#     # Example configuration
#     config = {
#         'world_id': 9,
#         'scale_factor': 1.0,
#         'render': True,
#         'use_gui': False,
#         'vehicle': 'hmmwv',
#         'system': 'pid',
#         'max_time': 60.0,
#         'speed': 8.0
#     }
    
#     # Run a single experiment
#     result = single_experiment(config)
#     print("Experiment Result:", result)

import json
import pandas as pd
import numpy as np
import contextlib

def multi_experiment(base_config,
                     runs_per_world=10,
                     config_json_path=None,
                     csv_path=None):
    """
    对指定的世界 ID 进行重复实验，并汇总结果。

    :param base_config: dict, 单次实验的基础配置字典（不含 world_id）
    :param runs_per_world: int, 每个 world_id 重复实验的次数
    :param config_json_path: str, 保存 low/mid/high id 列表的 JSON 的路径
    :param csv_path: str or None, 如果提供，则会把结果写到该 CSV 文件
    :return: pandas.DataFrame, 包含所有实验结果
    """
    # if config_json_path is None or not os.path.isfile(config_json_path):
    #     raise FileNotFoundError(f"Cannot find config JSON at {config_json_path}")

    # # 1) 读取 JSON
    # with open(config_json_path, 'r') as f:
    #     labels = json.load(f)

    # low_ids = labels['difficulty'].get('low', [])
    # mid_ids = labels['difficulty'].get('mid', [])
    # high_ids = labels['difficulty'].get('high', [])

    # world_ids = sorted(low_ids + mid_ids + high_ids)

    world_ids = [i for i in range(1, 101)]  # 假设我们要测试所有 100 个世界

    records = []
    for world_id in world_ids:
        for run_idx in range(runs_per_world):
            # 将 world_id 注入配置
            config = base_config.copy()
            config['world_id'] = world_id

            # 静默运行：屏蔽所有 stdout/stderr
            with open(os.devnull, 'w') as devnull, \
                 contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                gym = TALGym(config)
                gym.initialize()
                t2g, success, rolls, pitches = gym.run()
            
            # 这里是每次实验新定义一个Gym因而success的获取没问题
            records.append({
                'world_id':      world_id,
                'run_idx':       run_idx,
                'time_to_goal':  t2g,
                'success':       success,
                'avg_roll_deg':  np.mean(rolls) if rolls else np.nan,
                'avg_pitch_deg': np.mean(pitches) if pitches else np.nan,
            })

            # 可以选择保留进度打印
            print(f"[world {world_id} run {run_idx:02d}] time={t2g}, success={success}")

    df = pd.DataFrame(records)

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved all results to {csv_path}")

    return df


if __name__ == "__main__":
    SetChronoDataDirectories()
    base_config = {
        'scale_factor': 1.0,
        'render': False,
        'use_gui': False,
        'vehicle': 'hmmwv',
        'system': 'tal',
        'max_time': 60.0,
        'speed': 10.0,
        'collect_trajectory': True,   # 多实验时一般关闭数据收集，免得文件爆炸
    }

    df = multi_experiment(
        base_config,
        runs_per_world=5,
        config_json_path='/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final/config_ids.json',
        csv_path='tal10_multi_experiment_results.csv'
    )
    print(df.groupby('world_id')['success'].mean())