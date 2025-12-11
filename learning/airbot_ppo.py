"""PyTorch PPO training for AirbotPlayPickCube using ManiSkill-style logic."""

import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import mediapy as media
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import tyro

from mujoco_playground import registry
from mujoco_playground._src import wrapper_torch
from mujoco_playground.config import manipulation_params


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


@dataclass
class Args:
  # experiment setup
  seed: int = 1
  cuda: bool = True
  torch_deterministic: bool = True
  log_interval: int = 1
  compile_agent: bool = True
  use_amp: bool = False

  # environment
  env_id: str = "AirbotPlayPickCube"
  num_envs: int = 64
  num_steps: int = 50
  action_repeat: int = 1
  episode_length: int = 150

  # PPO hyper-parameters (ManiSkill defaults)
  total_timesteps: int = 1_000_000
  learning_rate: float = 3e-4
  num_minibatches: int = 16
  update_epochs: int = 4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_coef: float = 0.2
  ent_coef: float = 0.0
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  target_kl: float = 0.1
  reward_scale: float = 1.0
  norm_adv: bool = True
  finite_horizon_gae: bool = False

  # evaluation and visualization
  eval_interval: int = 10
  num_eval_episodes: int = 5
  render_every: int = 2
  video_width: int = 640
  video_height: int = 480

  # checkpointing
  checkpoint_dir: str = "checkpoints"
  checkpoint_interval: int = 25
  run_name: str = "airbot_ppo"
  log_dir: str = "runs"

  # runtime derived
  batch_size: int = 0
  minibatch_size: int = 0
  num_iterations: int = 0


class Agent(nn.Module):
  def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...]):
    super().__init__()
    self.critic = nn.Sequential(
        layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0),
    )
    self.actor_mean = nn.Sequential(
        layer_init(nn.Linear(obs_dim, hidden_sizes[0])),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        nn.Tanh(),
        layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01),
    )
    self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

  def get_value(self, obs: torch.Tensor) -> torch.Tensor:
    return self.critic(obs)

  def get_action_and_value(
      self, obs: torch.Tensor, action: torch.Tensor | None = None
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_mean = self.actor_mean(obs)
    action_logstd = self.actor_logstd.expand_as(action_mean)
    action_std = torch.exp(action_logstd)
    dist = Normal(action_mean, action_std)
    if action is None:
      action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)
    entropy = dist.entropy().sum(-1)
    return action, log_prob, entropy, self.critic(obs)


class PlaygroundPPOEnv:
  """Adapts RSLRLBraxWrapper outputs to ManiSkill-style tensors."""

  def __init__(self, vec_env: wrapper_torch.RSLRLBraxWrapper, device: torch.device):
    self.vec_env = vec_env
    self.device = device

  def reset(self) -> torch.Tensor:
    obs_td = self.vec_env.reset()
    return obs_td["state"].to(self.device)

  def step(self, action: torch.Tensor):
    obs_td, reward, done, info = self.vec_env.step(action)
    obs = obs_td["state"].to(self.device)
    reward = reward.to(self.device).flatten()
    done = done.to(self.device).flatten()
    time_outs = info.get("time_outs")
    if time_outs is None:
      time_outs = torch.zeros_like(done)
    else:
      time_outs = time_outs.to(self.device).flatten()
    return obs, reward, done, time_outs, info


def build_args_from_config() -> Args:
  cfg = manipulation_params.brax_ppo_config("AirbotPlayPickCube")
  args = Args()
  args.env_id = cfg.get("env_name", args.env_id)
  args.num_envs = cfg.num_envs
  args.num_steps = cfg.unroll_length
  args.total_timesteps = cfg.num_timesteps
  args.learning_rate = cfg.learning_rate
  args.num_minibatches = cfg.num_minibatches
  args.update_epochs = cfg.num_updates_per_batch
  args.gamma = cfg.discounting
  args.ent_coef = cfg.entropy_cost
  args.reward_scale = cfg.get("reward_scaling", 1.0)
  args.action_repeat = cfg.action_repeat
  args.episode_length = cfg.episode_length
  args.batch_size = args.num_envs * args.num_steps
  args.minibatch_size = args.batch_size // args.num_minibatches
  args.num_iterations = args.total_timesteps // args.batch_size
  return args


def make_env(args: Args, device: torch.device) -> PlaygroundPPOEnv:
  env = registry.load(args.env_id)
  vec_env = wrapper_torch.RSLRLBraxWrapper(
      env,
      num_actors=args.num_envs,
      seed=args.seed,
      episode_length=args.episode_length,
      action_repeat=args.action_repeat,
  )
  return PlaygroundPPOEnv(vec_env, device)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    next_value: torch.Tensor,
    args: Args,
) -> Tuple[torch.Tensor, torch.Tensor]:
  advantages = torch.zeros_like(rewards)
  last_gae_lam = 0
  for t in reversed(range(args.num_steps)):
    if args.finite_horizon_gae:
      next_non_terminal = 1.0 - time_outs[t]
    else:
      next_non_terminal = 1.0 - dones[t]
    next_values = next_value if t == args.num_steps - 1 else values[t + 1]
    delta = rewards[t] + args.gamma * next_values.flatten() * next_non_terminal - values[t].flatten()
    last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
    advantages[t] = last_gae_lam
  returns = advantages + values
  return advantages, returns


def evaluate(agent: Agent, args: Args, device: torch.device) -> Dict[str, float]:
  """Runs deterministic evaluation rollouts and returns metrics."""

  eval_env = registry.load(args.env_id)
  vec_env = wrapper_torch.RSLRLBraxWrapper(
      eval_env,
      num_actors=args.num_eval_episodes,
      seed=args.seed + 123,
      episode_length=args.episode_length,
      action_repeat=args.action_repeat,
  )
  envs = PlaygroundPPOEnv(vec_env, device)
  with torch.inference_mode():
    obs = envs.reset()
  total_rewards = torch.zeros(args.num_eval_episodes, device=device)
  done_mask = torch.zeros(args.num_eval_episodes, device=device)
  success_counts = []

  for _ in range(args.episode_length):
    with torch.inference_mode():
      action_mean = agent.actor_mean(obs)
    obs, reward, done, time_out, info = envs.step(action_mean)
    effective_done = torch.maximum(done, time_out)
    total_rewards += reward * (1 - done_mask)
    done_mask = torch.maximum(done_mask, effective_done)
    if "log" in info and "last_episode_success_count" in info["log"]:
      success_counts.append(info["log"]["last_episode_success_count"])
    if done_mask.all():
      break

  metrics = {
      "eval/episode_reward": total_rewards.mean().item(),
      "eval/reward_std": total_rewards.std().item(),
  }
  if success_counts:
    metrics["eval/success_rate"] = float(np.mean(success_counts))
  return metrics


def render_policy(agent: Agent, args: Args, device: torch.device, video_path: str):
  """Runs a single rollout and renders it to a video for qualitative inspection."""

  env_cfg = manipulation_params.brax_ppo_config(args.env_id)
  eval_env = registry.load(args.env_id, config=env_cfg)

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(args.seed + 2025)
  state = jit_reset(rng)
  rollout = [state]

  is_dict_obs = isinstance(eval_env.observation_size, dict)
  obs = state.obs["state"] if is_dict_obs else state.obs
  obs_torch = wrapper_torch._jax_to_torch(obs).to(device)
  if obs_torch.ndim == 1:
    obs_torch = obs_torch.unsqueeze(0)

  for _ in range(args.episode_length):
    with torch.inference_mode():
      action = agent.actor_mean(obs_torch)
    action = torch.clip(action, -1.0, 1.0)
    state = jit_step(state, wrapper_torch._torch_to_jax(action))
    rollout.append(state)
    obs = state.obs["state"] if is_dict_obs else state.obs
    obs_torch = wrapper_torch._jax_to_torch(obs).to(device)
    if obs_torch.ndim == 1:
      obs_torch = obs_torch.unsqueeze(0)
    if bool(np.array(state.done).item()):
      break

  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  render_every = max(1, args.render_every)
  fps = 1.0 / eval_env.dt / render_every
  traj = rollout[::render_every]
  frames = eval_env.render(
      traj,
      height=args.video_height,
      width=args.video_width,
      scene_option=scene_option,
  )
  media.write_video(video_path, frames, fps=fps)
  print(f"Rollout video saved as '{video_path}'.")


def save_checkpoint(
    agent: Agent,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: Args,
    run_dir: str,
    iter_idx: int,
    global_step: int,
    is_final: bool = False,
):
  os.makedirs(run_dir, exist_ok=True)
  ckpt_name = "final_ckpt.pt" if is_final else f"iter_{iter_idx + 1}.pt"
  ckpt_path = os.path.join(run_dir, ckpt_name)
  torch.save(
      {
          "agent": agent.state_dict(),
          "optimizer": optimizer.state_dict(),
          "scaler": scaler.state_dict(),
          "args": args,
          "iter_idx": iter_idx,
          "global_step": global_step,
      },
      ckpt_path,
  )
  tag = "final" if is_final else f"iteration {iter_idx + 1}"
  print(f"Saved {tag} checkpoint to {ckpt_path}")


def train(args: Args):
  if args.torch_deterministic:
    torch.backends.cudnn.deterministic = True
  else:
    torch.backends.cudnn.benchmark = True
  torch.set_float32_matmul_precision("high")

  device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
  device_type = device.type

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  run_dir = os.path.join(args.checkpoint_dir, args.run_name or args.env_id)

  envs = make_env(args, device)
  obs = envs.reset()
  obs_dim = obs.shape[1]
  action_dim = envs.vec_env.num_actions

  agent = Agent(obs_dim, action_dim, hidden_sizes=(256, 256)).to(device)
  if args.compile_agent and hasattr(torch, "compile"):
    agent = torch.compile(agent)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
  scaler = torch.amp.GradScaler(device_type=device_type, enabled=args.use_amp and device_type == "cuda")

  writer = SummaryWriter(os.path.join(args.log_dir, args.run_name or args.env_id))
  writer.add_text("config/args", str(args))

  # rollout storage
  obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
  actions_buf = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
  logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
  rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
  dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
  values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
  timeouts_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

  global_step = 0
  for iter_idx in range(args.num_iterations):
    policy_losses = []
    value_losses_list = []
    entropy_losses = []
    approx_kls = []
    clip_fracs = []
    start_time = time.time()
    for step in range(args.num_steps):
      global_step += args.num_envs
      obs_buf[step] = obs
      dones_buf[step] = torch.zeros(args.num_envs, device=device)

      with torch.inference_mode():
        action, log_prob, _, value = agent.get_action_and_value(obs)
      actions_buf[step] = action
      logprobs_buf[step] = log_prob
      values_buf[step] = value.flatten()

      next_obs, reward, done, time_out, info = envs.step(action)
      rewards_buf[step] = reward * args.reward_scale
      dones_buf[step] = done
      timeouts_buf[step] = time_out
      obs = next_obs

      if global_step % (args.num_envs * args.num_steps) == 0:
        success_rate = info.get("log", {}).get("last_episode_success_count")
        if success_rate is not None:
          print(f"Iter {iter_idx} step {global_step}: success {success_rate:.3f}")
          writer.add_scalar("train/success_rate", success_rate, global_step)

    with torch.no_grad():
      next_value = agent.get_value(obs).reshape(1, -1)
    advantages, returns = compute_gae(
        rewards_buf, values_buf, dones_buf, timeouts_buf, next_value, args
    )
    if args.norm_adv:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    b_obs = obs_buf.reshape(-1, obs_dim)
    b_actions = actions_buf.reshape(-1, action_dim)
    b_logprobs = logprobs_buf.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values_buf.reshape(-1)

    inds = torch.randperm(args.batch_size)
    for epoch in range(args.update_epochs):
      for start in range(0, args.batch_size, args.minibatch_size):
        end = start + args.minibatch_size
        mb_inds = inds[start:end]

        with torch.amp.autocast(device_type, enabled=args.use_amp and device_type == "cuda"):
          _, new_logprob, entropy, new_value = agent.get_action_and_value(
              b_obs[mb_inds], b_actions[mb_inds]
          )
        log_ratio = new_logprob - b_logprobs[mb_inds]
        ratio = log_ratio.exp()

        mb_advantages = b_advantages[mb_inds]
        if args.norm_adv:
          mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        policy_losses.append(pg_loss.detach())

        new_value = new_value.view(-1)
        if args.clip_coef > 0:
          value_clipped = b_values[mb_inds] + torch.clamp(
              new_value - b_values[mb_inds], -args.clip_coef, args.clip_coef
          )
          value_losses = (new_value - b_returns[mb_inds]).pow(2)
          value_losses_clipped = (value_clipped - b_returns[mb_inds]).pow(2)
          v_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
          v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
        value_losses_list.append(v_loss.detach())

        entropy_loss = entropy.mean()
        entropy_losses.append(entropy_loss.detach())
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
        approx_kl = ((ratio - 1) - log_ratio).mean()
        approx_kls.append(approx_kl.item())

      if approx_kls and approx_kls[-1] > args.target_kl:
        break

    explained_var = (
        1 - (returns - values_buf).pow(2).mean() / returns.var()
    ).item()

    metrics = {
        "loss/policy": torch.stack(policy_losses).mean().item() if policy_losses else 0.0,
        "loss/value": torch.stack(value_losses_list).mean().item() if value_losses_list else 0.0,
        "loss/entropy": torch.stack(entropy_losses).mean().item() if entropy_losses else 0.0,
        "train/approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "train/clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "train/value_explained_var": explained_var,
        "train/throughput_steps_per_s": args.batch_size / max(time.time() - start_time, 1e-6),
        "train/return_mean": rewards_buf.sum(dim=0).mean().item(),
    }

    for key, value in metrics.items():
      writer.add_scalar(key, value, global_step)

    if (iter_idx + 1) % args.log_interval == 0:
      metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
      print(f"Iteration {iter_idx + 1}/{args.num_iterations} | {metric_str}")

    if (iter_idx + 1) % args.eval_interval == 0:
      eval_metrics = evaluate(agent, args, device)
      eval_str = " | ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
      print(f"Eval @ iter {iter_idx + 1}: {eval_str}")
      for key, value in eval_metrics.items():
        writer.add_scalar(key, value, global_step)

    if args.checkpoint_interval > 0 and (iter_idx + 1) % args.checkpoint_interval == 0:
      save_checkpoint(agent, optimizer, scaler, args, run_dir, iter_idx, global_step)

  render_policy(agent, args, device, "airbot_rollout.mp4")
  save_checkpoint(agent, optimizer, scaler, args, run_dir, iter_idx, global_step, is_final=True)
  writer.flush()
  writer.close()


if __name__ == "__main__":
  default_args = build_args_from_config()
  cli_args = tyro.cli(Args, default=default_args)
  if cli_args.batch_size == 0:
    cli_args.batch_size = cli_args.num_envs * cli_args.num_steps
  if cli_args.minibatch_size == 0:
    cli_args.minibatch_size = cli_args.batch_size // cli_args.num_minibatches
  if cli_args.num_iterations == 0:
    cli_args.num_iterations = cli_args.total_timesteps // cli_args.batch_size
  train(cli_args)
