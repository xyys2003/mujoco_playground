#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vision-based PPO entrypoint for Panda tasks using Brax training APIs.

This script mirrors the training setup from ``training_vision_2.ipynb``
and ManiSkill's RGB PPO example, exposing a simple CLI to launch
pixel-based PPO on MuJoCo Playground Panda environments.
"""

import functools
import os
from dataclasses import dataclass
from typing import Dict, Optional

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from flax import linen, serialization
import jax
import mediapy as media
import numpy as np
import tyro

from mujoco_playground import manipulation, wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import (
    randomize_vision as randomize,
)
from mujoco_playground.config import manipulation_params


@dataclass
class Args:
    """Command line options for RGB PPO training."""

    env_id: str = "PandaPickCubeCartesian"
    num_envs: int = 1024
    total_timesteps: int = 5_000_000
    seed: int = 0

    render_width: int = 64
    render_height: int = 64
    render_batch_size: Optional[int] = None
    use_rasterizer: bool = False

    box_init_range: float = 0.1
    action_history_length: int = 5
    success_threshold: float = 0.03

    capture_video: bool = False
    video_path: Optional[str] = "rgb_ppo_eval.mp4"
    video_num_steps: int = 200
    video_render_every: int = 1

    checkpoint_path: Optional[str] = "rgb_ppo_params.msgpack"
    render_only: bool = False


def main() -> None:
    args = tyro.cli(Args)

    # Keep XLA from preallocating all GPU memory so Madrona can reserve buffers.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    env, env_cfg, config_overrides, episode_length = build_training_env(args)
    network_factory = build_network_factory()

    if args.render_only:
        params = load_checkpoint(args.checkpoint_path)
        make_inference_fn = ppo_networks.make_inference_fn(
            network_factory(env.observation_size, env.action_size)
        )
        maybe_render(args, env_cfg, config_overrides, make_inference_fn, params)
        return

    ppo_params = manipulation_params.brax_vision_ppo_config(args.env_id)
    ppo_params.num_timesteps = args.total_timesteps
    ppo_params.num_envs = args.num_envs
    ppo_params.num_eval_envs = args.num_envs
    ppo_params.network_factory = network_factory

    print(
        f"Launching RGB PPO for {args.env_id}: envs={args.num_envs}, "
        f"timesteps={args.total_timesteps}, seed={args.seed}"
    )

    train_fn = functools.partial(
        ppo.train,
        augment_pixels=True,
        **dict(ppo_params),
    )

    make_inference_fn, params, metrics = train_fn(environment=env)
    print("Training complete.")
    print(metrics)

    if args.checkpoint_path:
        save_checkpoint(params, args.checkpoint_path)

    maybe_render(args, env_cfg, config_overrides, make_inference_fn, params)


def build_training_env(args: Args):
    env_cfg = manipulation.get_default_config(args.env_id)
    episode_length = int(4 / env_cfg.ctrl_dt)

    config_overrides: Dict[str, object] = {
        "episode_length": episode_length,
        "vision": True,
        "obs_noise.brightness": [0.75, 2.0],
        "vision_config.use_rasterizer": args.use_rasterizer,
        "vision_config.render_batch_size": args.render_batch_size or args.num_envs,
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
    randomization_fn = functools.partial(randomize.domain_randomize, num_worlds=args.num_envs)
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=args.num_envs,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=randomization_fn,
    )

    return env, env_cfg, config_overrides, episode_length


def build_network_factory():
    return functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        policy_hidden_layer_sizes=[256, 256],
        value_hidden_layer_sizes=[256, 256],
        activation=linen.relu,
        normalise_channels=True,
    )


def maybe_render(
    args: Args,
    base_config,
    base_overrides: Dict[str, object],
    make_inference_fn,
    params,
) -> None:
    if not (args.capture_video and args.video_path):
        return

    render_policy_video(
        args=args,
        base_config=base_config,
        base_overrides=base_overrides,
        make_inference_fn=make_inference_fn,
        params=params,
    )


def render_policy_video(
    *,
    args: Args,
    base_config,
    base_overrides: Dict[str, object],
    make_inference_fn,
    params,
) -> None:
    """Roll out a trained policy and save RGB video."""

    eval_overrides = dict(base_overrides)
    eval_overrides.update(
        {
            "vision_config.render_batch_size": 1,
            "vision_config.render_width": args.render_width,
            "vision_config.render_height": args.render_height,
        }
    )

    eval_env = manipulation.load(
        args.env_id,
        config=manipulation.get_default_config(args.env_id),
        config_overrides=eval_overrides,
    )
    eval_env = wrapper.wrap_for_brax_training(
        eval_env,
        vision=True,
        num_vision_envs=1,
        episode_length=int(4 / base_config.ctrl_dt),
        action_repeat=1,
        randomization_fn=None,
    )

    policy = build_policy_fn(make_inference_fn, params)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(args.seed + 7)
    state = jit_reset(rng)
    rollout = [state]

    max_steps = min(args.video_num_steps, eval_env.unwrapped._config.episode_length)  # noqa: SLF001
    render_every = max(1, args.video_render_every)

    for _ in range(max_steps):
        obs = state.obs
        rng, action_key = jax.random.split(rng)
        action = policy(obs, action_key)
        state = jit_step(state, action)
        rollout.append(state)

    frames = eval_env.render(
        rollout[::render_every],
        width=args.render_width,
        height=args.render_height,
    )
    fps = 1.0 / eval_env.dt / render_every if hasattr(eval_env, "dt") else 30

    os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
    media.write_video(args.video_path, frames, fps=fps)
    print(f"Saved evaluation video to {args.video_path}")


def build_policy_fn(make_inference_fn, params):
    """Creates a callable that accepts (obs, key) regardless of source factory."""

    try:
        policy = make_inference_fn(params, deterministic=True)
    except TypeError:
        policy = make_inference_fn(params)

    def wrapped_policy(obs, key):
        try:
            return policy(obs, key)[0]
        except TypeError:
            return policy(obs)

    return wrapped_policy


def save_checkpoint(params, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    print(f"Saved params to {path}")


def load_checkpoint(path: Optional[str]):
    if not path:
        raise ValueError("--checkpoint_path is required when --render_only is set")
    with open(path, "rb") as f:
        data = f.read()
    params = serialization.msgpack_restore(data)
    print(f"Loaded params from {path}")
    return params


if __name__ == "__main__":
    main()
