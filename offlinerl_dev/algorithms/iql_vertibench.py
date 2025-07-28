"""
IQL adapted for VertiBench custom environment and offline dataset.
This script adapts the standard IQL algorithm to work with:
1. Custom VertiBench environment (off_road_art)
2. Custom offline dataset (HDF5 format from process_trajectories_for_offline_rl.py)
3. Custom observation/action/reward structure
"""

import copy
import os
import random
import uuid
import h5py
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import VertiBench environment
import sys
sys.path.append('/home/zkr/Documents/verti_bench')
from rl.off_road_VertiBench_offlinerl import off_road_art
from offlinerl_dev.modules.buffer import VertiBenchReplayBuffer


TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class VertiBenchTrainConfig:
    # Experiment
    device: str = "cuda"
    env_id: str = "VertiBench"  # Custom environment identifier
    world_id: int = 5  # VertiBench world configuration
    scale_factor: float = 1.0  # VertiBench scale factor
    dataset_path: str = "/home/zkr/Documents/verti_bench/offline_rl_dataset"  # Path to HDF5 dataset
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 5  # How many episodes run during evaluation
    max_timesteps: int = int(1e5)  # Max time steps to run environment
    checkpoints_path: Optional[str] = "/home/zkr/Documents/verti_bench/checkpoints"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    vf_lr: float = 1e-4  # V function learning rate
    qf_lr: float = 1e-4  # Critic learning rate
    actor_lr: float = 1e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Dropout for policy network
    # Wandb logging
    project: str = "VertiBench-OfflineRL"
    group: str = "IQL-VertiBench"
    name: str = "IQL-VertiBench"

    def __post_init__(self):
        self.name = f"{self.name}-world{self.world_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save("checkpoints/*.pt")


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    """Wrap environment with normalization."""
    def normalize_state(state):
        # 如果是新版 Gym 那种 (obs, info) 的 tuple
        if isinstance(state, tuple) and len(state) == 2:
            obs, info = state
            obs = (obs - state_mean) / state_std
            return obs, info
        # 否则直接当成 ndarray
        return (state - state_mean) / state_std
    
    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    times_to_goal   = []
    successes       = []
    avg_pitches     = []
    avg_rolls       = []

    import json
    with open("/home/zkr/Documents/verti_bench/envs/data/BenchMaps/sampled_maps/Configs/Final/config_labels.json", "r") as f:
        groups = json.load(f)
    # difficulty_low          = groups["difficulty"]["low"]
    # difficulty_mid          = groups["difficulty"]["mid"]
    # n_low = n_episodes//2
    # n_mid = n_episodes- n_low
    # test_world_ids = difficulty_low[:n_low] + difficulty_mid[:n_mid]

    difficulty_mid          = groups["difficulty"]["mid"]
    test_world_ids = difficulty_mid[:n_episodes]
    test_world_ids.reverse()  # Reverse to match the order in which they were sampled

    # for _ in range(n_episodes):
    for test_world_id in test_world_ids:
        pitch_list = []
        roll_list  = []
        # state, info = env.reset()
        state, info = env.full_reset(world_id=test_world_id)
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            pitch_list.extend(info.get('pitch_angles', []))
            roll_list.extend( info.get('roll_angles',  []))
        episode_rewards.append(episode_reward)
        times_to_goal.append(info.get('time_to_goal') or np.nan)
        successes.append(    info.get('success',    False))
        avg_pitches.append(np.mean(pitch_list) if pitch_list else np.nan)
        avg_rolls.append(  np.mean(roll_list)  if roll_list  else np.nan)
        
        print(f"Episode finished with reward: {episode_reward:.2f}, "
                f"time to goal: {info.get('time_to_goal') or np.nan:.2f}, "
                f"success: {info.get('success', False)}, "
                f"avg pitch: {np.mean(pitch_list) if pitch_list else np.nan:.2f}, "
                f"avg roll: {np.mean(roll_list) if roll_list else np.nan:.2f}")

    actor.train()
    return (
        np.asarray(episode_rewards),
        np.asarray(times_to_goal),
        np.asarray(successes),
        np.asarray(avg_pitches),
        np.asarray(avg_rolls),
    )


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward_vertibench(dataset, max_episode_steps=1000):
    """Modify rewards for VertiBench dataset."""
    # For VertiBench, we might want to normalize rewards based on the reward range
    min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    print(f"Original reward range: [{min_ret:.3f}, {max_ret:.3f}]")
    
    # Optional: normalize rewards
    if max_ret - min_ret > 0:
        dataset["rewards"] = (dataset["rewards"] - min_ret) / (max_ret - min_ret)
        # Verify normalization
        new_min = dataset["rewards"].min()
        new_max = dataset["rewards"].max()
        print(f"Normalized reward range: [{new_min:.3f}, {new_max:.3f}]")
    
    return dataset


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        log_dict["bc_loss"] = bc_losses.mean().item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: VertiBenchTrainConfig):
    """Train IQL on VertiBench environment and dataset."""
    
    # Create VertiBench environment
    print(f"Creating VertiBench environment with world_id={config.world_id}, scale_factor={config.scale_factor}")
    env = off_road_art(
        world_id=config.world_id,
        scale_factor=config.scale_factor,
        additional_render_mode='None'
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Environment specs:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Max action: {max_action}")

    # Create replay buffer and load dataset
    replay_buffer = VertiBenchReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    dataset = replay_buffer.load_vertibench_dataset(config.dataset_path)

    if config.normalize_reward:
        dataset = modify_reward_vertibench(dataset)
        # 同步归一化奖励到replay buffer
        replay_buffer._rewards[:len(dataset["rewards"])] = torch.tensor(
            dataset["rewards"].reshape(-1, 1), dtype=torch.float32, device=config.device
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    # Create networks
    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    
    # Create optimizers
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    # Initialize IQL trainer
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL on VertiBench, World: {config.world_id}, Seed: {seed}")
    print("---------------------------------------")

    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")

            # Now eval_actor returns five arrays:
            # (rewards, times_to_goal, successes, avg_pitches, avg_rolls)
            eval_rewards, eval_times, eval_successes, eval_pitches, eval_rolls = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )

            # Compute whatever you want to track
            reward_mean      = np.mean(eval_rewards)
            time_mean        = np.nanmean(eval_times)
            success_rate     = np.mean(eval_successes)
            pitch_mean       = np.nanmean(eval_pitches)
            roll_mean        = np.nanmean(eval_rolls)

            evaluations.append(reward_mean)

            print("---------------------------------------")
            print(f"Eval over {config.n_episodes} eps:")
            print(f"  reward   : {reward_mean:.3f}")
            print(f"  time     : {time_mean:.3f}")
            print(f"  success  : {success_rate*100:.1f}%")
            print(f"  pitch ⌀  : {pitch_mean:.3f}")
            print(f"  roll  ⌀  : {roll_mean:.3f}")
            print("---------------------------------------")

            # checkpointing
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            # log them all to WandB
            wandb.log({
                "eval/reward_mean"     : reward_mean,
                "eval/time_mean"       : time_mean,
                "eval/success_rate"    : success_rate,
                "eval/pitch_mean"      : pitch_mean,
                "eval/roll_mean"       : roll_mean,
            }, step=trainer.total_it)

    return trainer, evaluations


if __name__ == "__main__":
    train()
