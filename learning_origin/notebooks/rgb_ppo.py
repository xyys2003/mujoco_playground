#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vision-based PPO with PyTorch using MuJoCo Playground wrappers.

This script mirrors the ManiSkill PPO RGB structure and reuses the Torch
wrappers from ``wrapper_torch`` (as used in ``state_ppo.py``) to train a pixel
policy on Playground Panda tasks. Rendering is performed via the Playground
renderer after training.
"""

from __future__ import annotations

import functools
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import gymnasium as gym
import jax
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import tyro

from mujoco_playground import manipulation
from mujoco_playground._src.manipulation.franka_emika_panda import (
    randomize_vision as randomize,
)
from mujoco_playground._src.wrapper_torch import RSLRLBraxWrapper


def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    track: bool = False
    wandb_project_name: str = "MuJoCoPlayground"
    wandb_entity: Optional[str] = None

    capture_video: bool = False
    video_path: Optional[str] = None
    video_num_steps: int = 200
    video_render_every: int = 1
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None

    # Algorithm specific arguments
    env_id: str = "PandaPickCubeCartesian"
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_envs: int = 64
    num_eval_envs: int = 8

    num_steps: int = 50
    num_eval_steps: int = 50

    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    eval_freq: int = 10

    # Vision-specific
    render_width: int = 84
    render_height: int = 84
    render_batch_size: Optional[int] = None
    use_rasterizer: bool = False
    box_init_range: float = 0.1
    action_history_length: int = 5
    success_threshold: float = 0.03


class PlaygroundPPOEnv:
    """Small shim to expose Gym-like VectorEnv API on the Torch wrapper."""

    def __init__(self, vec_env: RSLRLBraxWrapper):
        self._env = vec_env
        self.num_envs = vec_env.num_envs

        raw_shape = vec_env.num_obs
        self.image_obs = isinstance(raw_shape, (tuple, list)) and len(raw_shape) >= 3
        if self.image_obs:
            self.obs_shape = tuple(raw_shape)
            obs_dim = int(np.prod(self.obs_shape))
        else:
            self.obs_shape = (int(np.prod(raw_shape)),)
            obs_dim = self.obs_shape[0]

        self.critic_obs_dim = vec_env.num_privileged_obs
        act_dim = int(np.prod(vec_env.num_actions))

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.obs_shape,
            dtype=np.float32,
        )

        critic_dim = self.critic_obs_dim or obs_dim
        self.single_critic_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(critic_dim,),
            dtype=np.float32,
        )

        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

    def _split_obs(self, obs_td):
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


class CNNEncoder(nn.Module):
    def __init__(self, input_shape: Sequence[int], hidden_size: int = 256):
        super().__init__()
        c, h, w = input_shape
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2)),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.cnn(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flat, hidden_size)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 if x.dtype in (torch.uint8, torch.int32, torch.int64) else x
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Agent(nn.Module):
    def __init__(self, envs: PlaygroundPPOEnv):
        super().__init__()
        self.image_obs = len(envs.single_observation_space.shape) == 3
        obs_shape = envs.single_observation_space.shape
        critic_obs_dim = int(np.prod(envs.single_critic_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        if self.image_obs:
            h, w, c = obs_shape
            self.encoder = CNNEncoder((c, h, w), hidden_size=256)
            feature_dim = 256
        else:
            self.encoder = nn.Identity()
            feature_dim = obs_shape[0]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        if self.image_obs and obs.dim() == 4:
            obs = obs.permute(0, 3, 1, 2)
        return self.encoder(obs)

    def get_value(self, obs):
        return self.critic(self._encode(obs))

    def get_action(self, obs, deterministic: bool = False):
        features = self._encode(obs)
        action_mean = self.actor_mean(features)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs, critic_obs=None, action=None):
        obs_features = self._encode(obs)
        if critic_obs is not None and critic_obs.shape != obs_features.shape:
            critic_features = critic_obs
        else:
            critic_features = obs_features
        action_mean = self.actor_mean(obs_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(critic_features)
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


def preprocess_obs(obs: torch.Tensor, envs: PlaygroundPPOEnv) -> torch.Tensor:
    return obs.float()


def make_playground_vec_env(args: Args, num_envs: int, seed: int):
    env_cfg = manipulation.get_default_config(args.env_id)
    episode_length = int(4 / env_cfg.ctrl_dt)
    config_overrides: Dict[str, object] = {
        "episode_length": episode_length,
        "vision": True,
        "obs_noise.brightness": [0.75, 2.0],
        "vision_config.use_rasterizer": args.use_rasterizer,
        "vision_config.render_batch_size": args.render_batch_size or num_envs,
        "vision_config.render_width": args.render_width,
        "vision_config.render_height": args.render_height,
        "box_init_range": args.box_init_range,
        "action_history_length": args.action_history_length,
        "success_threshold": args.success_threshold,
    }

    env = manipulation.load(
        args.env_id,
        config=env_cfg,
        config_overrides=config_overrides,
    )
    randomization_fn = functools.partial(randomize.domain_randomize, num_worlds=num_envs)
    vec_env = RSLRLBraxWrapper(
        env=env,
        num_actors=num_envs,
        seed=seed,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=randomization_fn,
        render_callback=None,
        device_rank=None,
    )

    ppo_env = PlaygroundPPOEnv(vec_env)
    return env, env_cfg, config_overrides, ppo_env


class RolloutVideoRenderer:
    def __init__(self, env, env_cfg, args: Args):
        self.env = env
        self.env_cfg = env_cfg
        self.args = args
        self.jit_reset = jax.jit(env.reset)
        self.jit_step = jax.jit(env.step)

    def render(self, agent: Agent, device: torch.device):
        if not (self.args.capture_video and self.args.video_path):
            return
        rng = jax.random.PRNGKey(self.args.seed + 2025)
        state = self.jit_reset(rng)
        rollout = [state]
        max_steps = min(self.args.video_num_steps, self.env_cfg.episode_length)
        render_every = max(1, self.args.video_render_every)

        agent.eval()
        for _ in range(max_steps):
            obs = state.obs["state"] if isinstance(state.obs, dict) else state.obs
            obs_tensor = torch.from_numpy(np.array(obs)).to(device).float().unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_tensor, deterministic=True).cpu().numpy()[0]
            state = self.jit_step(state, action)
            rollout.append(state)

        frames = self.env.render(rollout[::render_every], width=self.args.render_width, height=self.args.render_height)
        fps = 1.0 / self.env.dt / render_every if hasattr(self.env, "dt") else 30
        os.makedirs(os.path.dirname(self.args.video_path) or ".", exist_ok=True)
        media.write_video(self.args.video_path, frames, fps=fps)
        print(f"Saved evaluation video to {self.args.video_path}")


def main():
    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        run_name = f"{args.env_id}__ppo_rgb_torch__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    train_num_envs = args.num_envs if not args.evaluate else 1
    eval_num_envs = args.num_eval_envs

    train_env, env_cfg, config_overrides, envs = make_playground_vec_env(args, train_num_envs, seed=args.seed)
    _, _, _, eval_envs = make_playground_vec_env(args, eval_num_envs, seed=args.seed + 1)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    logger = None
    if not args.evaluate:
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

    obs_shape = envs.single_observation_space.shape
    critic_shape = envs.single_critic_observation_space.shape
    action_shape = envs.single_action_space.shape

    obs_buf = torch.zeros(
        (args.num_steps, train_num_envs) + obs_shape,
        device=device,
    )
    critic_obs_buf = torch.zeros(
        (args.num_steps, train_num_envs) + critic_shape,
        device=device,
    )
    actions_buf = torch.zeros(
        (args.num_steps, train_num_envs) + action_shape,
        device=device,
    )
    logprobs_buf = torch.zeros((args.num_steps, train_num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, train_num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, train_num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, train_num_envs), device=device)

    global_step = 0
    start_time = time.time()

    next_obs, info = envs.reset(seed=args.seed)
    next_obs = preprocess_obs(next_obs.to(device), envs)
    next_critic_obs = info["critic_obs"].to(device)

    eval_obs, eval_info = eval_envs.reset(seed=args.seed + 123)
    eval_obs = preprocess_obs(eval_obs.to(device), envs)
    eval_critic_obs = eval_info["critic_obs"].to(device)

    next_done = torch.zeros(train_num_envs, device=device)
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

    renderer = RolloutVideoRenderer(train_env, env_cfg, args)

    if args.evaluate:
        agent.eval()
        with torch.no_grad():
            eval_obs, eval_info = eval_envs.reset(seed=args.seed + 1234)
            eval_obs = preprocess_obs(eval_obs.to(device), envs)
            eval_critic_obs = eval_info["critic_obs"].to(device)
            eval_returns = torch.zeros(args.num_eval_steps, device=device)
            for eval_step in range(args.num_eval_steps):
                eval_action = agent.get_action(eval_obs, deterministic=True)
                eval_obs, eval_reward, eval_done, eval_trunc, eval_info = eval_envs.step(
                    clip_action(eval_action)
                )
                eval_obs = preprocess_obs(eval_obs.to(device), envs)
                eval_critic_obs = eval_info["critic_obs"].to(device)
                eval_returns[eval_step] = eval_reward.mean()
                if eval_done.sum() + eval_trunc.sum() > 0:
                    break
            mean_return = eval_returns[: eval_step + 1].mean().item()
            print(f"Eval return: {mean_return}")
        renderer.render(agent, device)
        return

    for iteration in range(1, args.num_iterations + 1):
        final_values = torch.zeros((args.num_steps, train_num_envs), device=device)
        agent.eval()
        rollout_logs: dict[str, list[float]] = defaultdict(list)

        if iteration % args.eval_freq == 1:
            eval_obs, eval_info = eval_envs.reset(seed=args.seed + 1234)
            eval_obs = preprocess_obs(eval_obs.to(device), envs)
            eval_critic_obs = eval_info["critic_obs"].to(device)
            with torch.no_grad():
                eval_returns = torch.zeros(args.num_eval_steps, device=device)
                for eval_step in range(args.num_eval_steps):
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, eval_reward, eval_done, eval_trunc, eval_info = eval_envs.step(clip_action(eval_action))
                    eval_obs = preprocess_obs(eval_obs.to(device), envs)
                    eval_critic_obs = eval_info["critic_obs"].to(device)
                    eval_returns[eval_step] = eval_reward.mean()
                    if eval_done.sum() + eval_trunc.sum() > 0:
                        break
                mean_return = eval_returns[: eval_step + 1].mean().item()
                print(f"Eval return: {mean_return}")
                if logger is not None:
                    logger.add_scalar("charts/eval_return", mean_return, global_step)

        agent.train()
        for step in range(args.num_steps):
            global_step += train_num_envs

            obs_buf[step] = next_obs
            critic_obs_buf[step] = next_critic_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_critic_obs)
                values_buf[step] = value.flatten()

            clipped_action = clip_action(action)
            actions_buf[step] = clipped_action
            logprobs_buf[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(clipped_action)
            next_obs = preprocess_obs(next_obs.to(device), envs)
            next_critic_obs = info["critic_obs"].to(device)

            done = torch.logical_or(terminated.bool(), truncated.bool()).float()
            dones_buf[step] = done
            rewards_buf[step] = reward
            next_done = done

            rollout_logs["reward"].append(reward.mean().item())
            for key, value in info["log"].items():
                rollout_logs[key].append(value)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            final_values[-1] = next_value
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                nextnonterminal = 1.0 - dones_buf[t]
                nextvalues = final_values[t]
                delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], None, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
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

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            if rollout_logs:
                for key, values in rollout_logs.items():
                    logger.add_scalar(f"rollout/{key}", np.mean(values), global_step)

        if args.save_model and iteration % args.eval_freq == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"{run_name}_iter{iteration}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        print("Iteration done: ", iteration)
        if logger is not None:
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    renderer.render(agent, device)

    if logger is not None:
        logger.close()

    if args.save_model:
        final_path = args.checkpoint or os.path.join("checkpoints", f"{run_name}_final.pt")
        os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
        torch.save(agent.state_dict(), final_path)
        print(f"Saved final policy to {final_path}")


if __name__ == "__main__":
    main()
