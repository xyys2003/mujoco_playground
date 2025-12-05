#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script converted from manipulation.ipynb, modified to use
PyTorch PPO (ported from ManiSkill PPO implementation) on MuJoCo Playground.

Original notebook: MuJoCo Playground manipulation example.
"""

# ===== Cell 4 =====
# @title Install pre-requisites
# !pip install mujoco  # (original notebook shell command)
# !pip install mujoco_mjx  # (original notebook shell command)
# !pip install brax  # (original notebook shell command)

# ===== Cell 5 =====
# @title Check if MuJoCo installation was successful

import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_EXTERNAL_PLATFORM_CONFIG_DIR",
                      "/data/yufei/egl_external_platform.d")

if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError(
        'Cannot communicate with GPU. '
        'Make sure you are using a GPU Colab runtime. '
        'Go to the Runtime menu and select Choose runtime type.'
    )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
EGL_EXTERNAL_PLATFORM_CONFIG = r"""{
  "file_format_version" : "1.0.0",
  "ICD" : {
    "library_path" : "libEGL_nvidia.so.0"
  }
}"""

custom_egl_dir = "/data/yufei/egl_external_platform.d"
os.environ['__EGL_EXTERNAL_PLATFORM_CONFIG_DIR'] = custom_egl_dir

egl_external_platform_config_path = os.path.join(
    custom_egl_dir,
    "10_nvidia.json"
)

os.makedirs(os.path.dirname(egl_external_platform_config_path), exist_ok=True)
with open(egl_external_platform_config_path, "w") as f:
    f.write(EGL_EXTERNAL_PLATFORM_CONFIG)

print("EGL config written to:", egl_external_platform_config_path)

import mujoco

try:
    mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
    raise e from RuntimeError(
        'Something went wrong during installation. Check the shell output above '
        'for more information.\n'
        'If using a hosted Colab runtime, make sure you enable GPU acceleration '
        'by going to the Runtime menu and selecting "Choose runtime type".'
    )

print('MuJoCo installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# ===== Cell 6 =====
# @title Import packages for plotting and creating graphics
import time
from datetime import datetime
from typing import Any, Optional

import numpy as np
import mediapy as media
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, linewidth=100)

# ===== Cell 7 =====
# @title Import MuJoCo, MJX, and JAX
import jax
from jax import numpy as jp

print("JAX devices:", jax.devices())

# ===== Cell 9 =====
# @title Import The Playground

from mujoco_playground import wrapper,wrapper_torch  # this is the Brax-style wrapper file
from mujoco_playground import registry
from mujoco_playground._src.wrapper_torch import RSLRLBraxWrapper
# ===== Environment setup =====

env_name = 'AirbotPlayPick'
env = registry.load(env_name)                     # MjxEnv (JAX)
env_cfg = registry.get_default_config(env_name)   # contains episode_length, action_repeat, etc.

print("Loaded MuJoCo Playground env:", env_name)
print("Env config:", env_cfg)

# ==============================================================
#           PyTorch PPO (ported from ManiSkill PPO)
# ==============================================================

import random
from dataclasses import dataclass
from collections import defaultdict

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import tyro

# NOTE: adjust this import to match where you put RSLRLBraxWrapper



# ---------- PPO Args ----------

@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = False
    wandb_project_name: str = "MuJoCoPlayground"
    wandb_entity: Optional[str] = None

    capture_video: bool = False    # 视频这边先关掉，后面你可以加
    video_path: Optional[str] = None
    video_num_steps: int = 200
    video_render_every: int = 1
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None

    # Algorithm specific arguments
    env_id: str = "AirbotPlayPick"  # 仅用于命名，不用来 gym.make
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 64
    num_eval_envs: int = 8

    num_steps: int = 50
    num_eval_steps: int = 50

    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 25
    finite_horizon_gae: bool = False
    success_once: bool = True

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------- Playground env adapter ----------

class PlaygroundPPOEnv:
    """
    Adapt MuJoCo Playground + RSLRLBraxWrapper to the ManiSkill PPO-style env
    interface.

    The wrapper flattens TensorDict observations into vanilla torch tensors so
    that the rest of this script can stay close to the ManiSkill PPO reference
    implementation. It also exposes privileged observations (if available) via
    ``info["critic_obs"]``.
    """

    def __init__(self, vec_env: RSLRLBraxWrapper):
        self._env = vec_env
        self.num_envs = vec_env.num_envs

        obs_dim = vec_env.num_obs
        self.critic_obs_dim = vec_env.num_privileged_obs
        act_dim = vec_env.num_actions

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        critic_dim = self.critic_obs_dim or obs_dim
        self.single_critic_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(critic_dim,),
            dtype=np.float32,
        )

        # RSLRLBraxWrapper internally clips actions to [-1, 1]
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

    def _split_obs(self, obs_td):
        """Return actor and critic observations from a TensorDict-like object.

        Some environments may return plain tensors instead of TensorDicts when
        asymmetric (privileged) observations are disabled. Be permissive here
        so we can gracefully handle both cases.
        """

        def _maybe_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        if hasattr(obs_td, "get"):
            obs = obs_td.get("state", None)
            critic_obs = obs_td.get("privileged_state", None)
        else:
            obs = None
            critic_obs = None

        if obs is None:
            try:
                obs = obs_td["state"]
            except Exception:
                obs = obs_td

        if critic_obs is None:
            try:
                critic_obs = obs_td["privileged_state"]
            except Exception:
                critic_obs = None

        obs = _maybe_tensor(obs).float()
        critic_obs = _maybe_tensor(critic_obs).float() if critic_obs is not None else obs
        return obs, critic_obs

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            # Currently unused, present for API parity.
            _ = seed
        obs_td = self._env.reset()
        obs, critic_obs = self._split_obs(obs_td)
        info = {"critic_obs": critic_obs}
        return obs, info

    def step(self, action: torch.Tensor):
        obs_td, reward, done, info = self._env.step(action)
        obs, critic_obs = self._split_obs(obs_td)
        time_outs = info.get("time_outs", torch.zeros_like(done))
        terminations = torch.logical_and(done.bool(), ~time_outs.bool()).float()
        truncations = time_outs.float()
        infos = {
            "critic_obs": critic_obs,
            "log": info.get("log", {}),
        }
        return obs, reward, terminations, truncations, infos

    def close(self):
        pass


def make_playground_vec_env(num_envs: int, seed: int) -> PlaygroundPPOEnv:
    vec_env = RSLRLBraxWrapper(
        env=env,  
        num_actors=num_envs,
        seed=seed,
        episode_length=env_cfg.episode_length,
        action_repeat=env_cfg.action_repeat,
        randomization_fn=None,
        render_callback=None,
        device_rank=None,  
    )

    ppo_env = PlaygroundPPOEnv(vec_env)
    return ppo_env

 
 
def record_policy_video(agent: nn.Module, args: Args, device: torch.device):
    if not args.capture_video or args.video_path is None:
        return

    # Follow manipulation.ipynb: roll out with the current policy and render frames.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed + 2025)
    state = jit_reset(rng)
    rollout = [state]

    max_steps = min(args.video_num_steps, env_cfg.episode_length)
    render_every = max(1, args.video_render_every)

    agent.eval()
    for _ in range(max_steps):
        obs = state.obs["state"] if isinstance(state.obs, dict) else state.obs
        obs_tensor = torch.from_numpy(np.array(obs)).to(device).float().unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(obs_tensor, deterministic=True).cpu().numpy()[0]
        state = jit_step(state, action)
        rollout.append(state)

    frames = env.render(rollout[::render_every])
    fps = 1.0 / env.dt / render_every if hasattr(env, "dt") else 30

    os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
    media.write_video(args.video_path, frames, fps=fps)
    print(f"Saved evaluation video to {args.video_path}")


# ---------- PPO Agent ----------

class Agent(nn.Module):
    def __init__(self, envs: PlaygroundPPOEnv):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        critic_obs_dim = int(np.prod(envs.single_critic_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        # logstd 初始为 -0.5 (跟 ManiSkill 脚本一致)
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, obs, critic_obs=None):
        critic_input = critic_obs if critic_obs is not None else obs
        return self.critic(critic_input)

    def get_action(self, obs, deterministic: bool = False):
        action_mean = self.actor_mean(obs)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs, critic_obs=None, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.get_value(obs, critic_obs)
        return action, log_prob, entropy, value


class Logger:
    def __init__(self, log_wandb: bool = False, tensorboard: Optional[SummaryWriter] = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
        self._wandb = None
        if self.log_wandb:
            import wandb
            self._wandb = wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb and self._wandb is not None:
            self._wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def main():
    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        run_name = f"{args.env_id}__ppo_torch__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    train_num_envs = args.num_envs if not args.evaluate else 1
    eval_num_envs = args.num_eval_envs

    envs = make_playground_vec_env(train_num_envs, seed=args.seed)
    eval_envs = make_playground_vec_env(eval_num_envs, seed=args.seed + 1)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    logger = None
    if not args.evaluate:
        print("Running training")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            )),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation only")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs_buf = torch.zeros(
        (args.num_steps, train_num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    critic_obs_buf = torch.zeros(
        (args.num_steps, train_num_envs)
        + envs.single_critic_observation_space.shape,
        device=device,
    )
    actions_buf = torch.zeros(
        (args.num_steps, train_num_envs) + envs.single_action_space.shape,
        device=device,
    )
    logprobs_buf = torch.zeros((args.num_steps, train_num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, train_num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, train_num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, train_num_envs), device=device)

    global_step = 0
    start_time = time.time()

    next_obs, info = envs.reset(seed=args.seed)
    next_obs = next_obs.to(device)
    next_critic_obs = info["critic_obs"].to(device)

    eval_obs, eval_info = eval_envs.reset(seed=args.seed + 123)
    eval_obs = eval_obs.to(device)
    eval_critic_obs = eval_info["critic_obs"].to(device)

    next_done = torch.zeros(train_num_envs, device=device)
    success_once_tracker = torch.zeros(train_num_envs, device=device, dtype=torch.bool)

    print("####")
    print(f"num_iterations={args.num_iterations}, num_envs={train_num_envs}, num_eval_envs={eval_num_envs}")
    print(f"minibatch_size={args.minibatch_size}, batch_size={args.batch_size}, update_epochs={args.update_epochs}")
    print("####")

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, train_num_envs), device=device)
        agent.eval()
        rollout_logs: dict[str, list[float]] = defaultdict(list)

        # Evaluation
        if iteration % args.eval_freq == 1:
            print("Evaluating...")
            eval_obs, eval_info = eval_envs.reset(seed=args.seed + 1234)
            eval_obs = eval_obs.to(device)
            eval_critic_obs = eval_info["critic_obs"].to(device)
            eval_returns = []
            eval_success_once = torch.zeros(eval_num_envs, device=device, dtype=torch.bool)
            eval_success_rates: list[float] = []

            with torch.no_grad():
                for _ in range(args.num_eval_steps):
                    actions_eval = agent.get_action(eval_obs, deterministic=True)
                    actions_eval = clip_action(actions_eval)
                    eval_obs, eval_rew, eval_terms, eval_truncs, eval_infos = eval_envs.step(actions_eval)
                    eval_obs = eval_obs.to(device)
                    eval_critic_obs = eval_infos["critic_obs"].to(device)
                    eval_rew = eval_rew.to(device)
                    eval_returns.append(eval_rew.mean().item())

                    eval_success_values = None
                    for k, v in eval_infos.get("log", {}).items():
                        if k in ("success", "is_success"):
                            if isinstance(v, torch.Tensor):
                                eval_success_values = v.to(device)
                            else:
                                try:
                                    eval_success_values = torch.as_tensor(v, device=device)
                                except Exception:
                                    eval_success_values = None
                        elif k == "reward/success":
                            if isinstance(v, torch.Tensor):
                                eval_success_values = (v > 0).to(device)
                            else:
                                try:
                                    eval_success_values = torch.as_tensor(v, device=device) > 0
                                except Exception:
                                    eval_success_values = None

                    eval_next_done = torch.logical_or(eval_terms.bool(), eval_truncs.bool())
                    if args.success_once:
                        if eval_success_values is not None:
                            eval_success_once |= eval_success_values.bool().view(-1)
                        if eval_next_done.any():
                            ended = eval_next_done.bool()
                            eval_success_rates.append(eval_success_once[ended].float().mean().item())
                            eval_success_once[ended] = False

            mean_return = float(np.mean(eval_returns)) if eval_returns else 0.0
            mean_success_once = float(np.mean(eval_success_rates)) if eval_success_rates else 0.0
            print(f"Eval mean reward: {mean_return:.3f}")
            if logger is not None:
                logger.add_scalar("eval/episode_reward", mean_return, global_step)
                if eval_success_rates:
                    logger.add_scalar("eval/success_once", mean_success_once, global_step)

            if args.evaluate:
                break

        # Save model periodically
        if args.save_model and iteration % args.eval_freq == 1 and not args.evaluate:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # LR annealing
        if args.anneal_lr and not args.evaluate:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        # -------- rollout --------
        for step in range(args.num_steps):
            global_step += train_num_envs
            obs_buf[step] = next_obs
            critic_obs_buf[step] = next_critic_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, critic_obs=next_critic_obs
                )
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_obs = next_obs.to(device)
            next_critic_obs = infos["critic_obs"].to(device)
            reward = reward.to(device)
            terminations = terminations.to(device)
            truncations = truncations.to(device)

            next_done = torch.logical_or(terminations.bool(), truncations.bool()).to(torch.float32)
            rewards_buf[step] = reward.view(-1) * args.reward_scale

            success_values = None
            for k, v in infos.get("log", {}).items():
                if isinstance(v, torch.Tensor):
                    rollout_logs[k].append(v.float().mean().item())
                else:
                    try:
                        rollout_logs[k].append(float(v))
                    except TypeError:
                        continue

                if k in ("success", "is_success"):
                    if isinstance(v, torch.Tensor):
                        success_values = v.to(device)
                    else:
                        try:
                            success_values = torch.as_tensor(v, device=device)
                        except Exception:
                            success_values = None
                elif k == "reward/success":
                    if isinstance(v, torch.Tensor):
                        success_values = (v > 0).to(device)
                    else:
                        try:
                            success_values = torch.as_tensor(v, device=device) > 0
                        except Exception:
                            success_values = None

            if args.success_once:
                if success_values is not None:
                    success_once_tracker |= success_values.bool().view(-1)
                if next_done.any():
                    ended = next_done.bool()
                    rollout_logs["success_once"].append(
                        success_once_tracker[ended].float().mean().item()
                    )
                    success_once_tracker[ended] = False

            if truncations.any():
                with torch.no_grad():
                    timeout_values = agent.get_value(
                        next_obs, critic_obs=next_critic_obs
                    ).flatten()
                    final_values[step] = final_values[step].masked_scatter(
                        truncations.bool(), timeout_values[truncations.bool()]
                    )


        rollout_time = time.time() - rollout_time

        # -------- GAE & returns --------
        with torch.no_grad():
            next_value = agent.get_value(next_obs, critic_obs=next_critic_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]

                real_next_values = next_not_done * nextvalues + final_values[t]
                if args.finite_horizon_gae:
                    if t == args.num_steps - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0

                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards_buf[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values_buf[t]
                else:
                    delta = rewards_buf[t] + args.gamma * real_next_values - values_buf[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                    )

            returns = advantages + values_buf

        # flatten batch
        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_critic_obs = critic_obs_buf.reshape(
            (-1,) + envs.single_critic_observation_space.shape
        )
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # -------- PPO update --------
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    critic_obs=b_critic_obs[mb_inds],
                    action=b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        # explained variance
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger is not None and not args.evaluate:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps)
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
            for log_key, values in rollout_logs.items():
                if values:
                    logger.add_scalar(f"env/{log_key}", float(np.mean(values)), global_step)

    # end main training loop

    if args.capture_video and args.video_path is not None:
        record_policy_video(agent, args, device)

    if not args.evaluate and args.save_model:
        final_model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    if logger is not None:
        logger.close()

    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main()
