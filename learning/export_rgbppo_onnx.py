#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export rgb_ppo checkpoint to ONNX.

Supports two signatures:
  1) rgb -> action
  2) (rgb, state) -> action

It auto-detects whether checkpoint contains a state branch by checking keys
like 'feat.state_fc.0.weight'. You can also force with --force-with-state / --force-no-state.
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_mani_wrapper import MJXManiLikeVectorEnv


# -------------------------
# Model (must match training)
# -------------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    def __init__(self, sample_obs: Dict[str, torch.Tensor]):
        super().__init__()
        if "rgb" not in sample_obs:
            raise KeyError("sample_obs must contain key 'rgb'")
        rgb = sample_obs["rgb"]
        if rgb.ndim != 4:
            raise ValueError(f"Expected rgb obs shape [B,H,W,C], got {tuple(rgb.shape)}")

        _, H, W, C = rgb.shape
        in_channels = int(C)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, int(H), int(W)), dtype=torch.float32)
            n_flat = int(self.cnn(dummy).shape[1])

        self.fc = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU())

        self.state_fc = None
        self.out_dim = 256
        if "state" in sample_obs:
            s = sample_obs["state"]
            if s.ndim != 2:
                raise ValueError(f"Expected state obs shape [B,D], got {tuple(s.shape)}")
            sdim = int(s.shape[-1])
            self.state_fc = nn.Sequential(nn.Linear(sdim, 256), nn.ReLU())
            self.out_dim += 256

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs["rgb"].float().permute(0, 3, 1, 2) / 255.0
        z = self.fc(self.cnn(x))
        if self.state_fc is not None and ("state" in obs):
            z2 = self.state_fc(obs["state"].float())
            z = torch.cat([z, z2], dim=1)
        return z


class Agent(nn.Module):
    def __init__(self, action_dim: int, sample_obs: Dict[str, torch.Tensor]):
        super().__init__()
        self.feat = NatureCNN(sample_obs)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.feat.out_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.feat.out_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)


# -------------------------
# Env + assets helpers (same as your eval)
# -------------------------
def _default_airbot_body_map(reso: str) -> Dict[str, str]:
    assets = mjx_env.ROOT_PATH / "manipulation/airbot_play/3dgs"
    r = str(reso)
    body = {
        "arm_base": (assets / r / "arm_base.ply").as_posix(),
        "link1": (assets / r / "link1.ply").as_posix(),
        "link2": (assets / r / "link2.ply").as_posix(),
        "link3": (assets / r / "link3.ply").as_posix(),
        "link4": (assets / r / "link4.ply").as_posix(),
        "link5": (assets / r / "link5.ply").as_posix(),
        "link6": (assets / r / "link6.ply").as_posix(),
        "left": (assets / r / "right.ply").as_posix(),
        "right": (assets / r / "left.ply").as_posix(),
        "box": (assets / "green_cube.ply").as_posix(),
    }
    return body


def _load_body_map_from_json(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("gs_body_gaussians_json must be a JSON dict: {body_name: ply_path}")
    return {str(k): str(v) for k, v in m.items()}


def _validate_plys(body_map: Dict[str, str], background_ply: Optional[str]):
    missing = []
    for k, p in body_map.items():
        if not os.path.exists(p):
            missing.append(f"{k}: {p}")
    if background_ply is not None and not os.path.exists(background_ply):
        missing.append(f"background: {background_ply}")
    if missing:
        raise FileNotFoundError("Missing PLY files:\n" + "\n".join(missing))


def _load_ckpt_any(path: str, device: torch.device) -> Tuple[Dict, Dict]:
    """
    Returns (train_args_dict, model_state_dict)
    - If ckpt is the full dict: use ckpt["args"], ckpt["model_state"]
    - If ckpt is a bare state_dict: train_args={}, model_state=ckpt
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and ("model_state" in obj or "args" in obj):
        train_args = obj.get("args", {}) or {}
        model_state = obj.get("model_state", None)
        if model_state is None:
            # allow weird case where dict is already a state_dict
            model_state = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        return train_args, model_state
    if isinstance(obj, dict):
        return {}, obj
    raise ValueError(f"Unsupported checkpoint format: {type(obj)}")


def _ckpt_has_state_branch(state_dict: Dict[str, torch.Tensor]) -> bool:
    # NatureCNN uses self.state_fc = nn.Sequential(...)
    # keys typically: "feat.state_fc.0.weight", "feat.state_fc.0.bias", ...
    for k in state_dict.keys():
        if "feat.state_fc" in k:
            return True
    return False


# -------------------------
# ONNX policy wrappers
# -------------------------
class PolicyRGB(nn.Module):
    """rgb -> action (deterministic mean)"""
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B,H,W,3] uint8/float
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        obs = {"rgb": rgb}
        z = self.agent.feat(obs)
        act = self.agent.actor_mean(z)
        return act


class PolicyRGBState(nn.Module):
    """(rgb, state) -> action (deterministic mean)"""
    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def forward(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        obs = {"rgb": rgb, "state": state}
        z = self.agent.feat(obs)
        act = self.agent.actor_mean(z)
        return act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str, help="Path to ckpt_xxxxx.pt or final_ckpt.pt")
    parser.add_argument("--output", required=True, type=str, help="Output ONNX path")
    parser.add_argument("--device-rank", type=int, default=0, help="CUDA device rank")
    parser.add_argument("--export-on-cpu", action="store_true", help="Move model to CPU for export")
    parser.add_argument("--force-with-state", action="store_true", help="Force export (rgb,state)->action")
    parser.add_argument("--force-no-state", action="store_true", help="Force export rgb->action (must match ckpt arch)")
    parser.add_argument("--gs-camera-ids", type=str, default=None, help="Comma-separated camera ids to override ckpt")
    parser.add_argument("--gs-camera-names", type=str, default=None, help="Comma-separated camera names to override ckpt")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and (not args.export_on_cpu)
    device = torch.device(f"cuda:{args.device_rank}" if use_cuda else "cpu")

    train_args, model_state = _load_ckpt_any(args.ckpt, device)
    if not isinstance(model_state, dict):
        raise ValueError("model_state is not a dict state_dict")

    # Infer env config from ckpt args (fallbacks are safe)
    env_name = str(train_args.get("env_name", "AirbotPlayPickCube"))
    seed = int(train_args.get("seed", 1))
    episode_seconds = float(train_args.get("episode_seconds", 3.0))
    ctrl_dt = float(train_args.get("ctrl_dt", 0.04))
    action_repeat = int(train_args.get("action_repeat", 1))
    include_state = bool(train_args.get("include_state", True))
    vision_backend = str(train_args.get("vision_backend", "gs")).lower().strip()

    if vision_backend != "gs":
        raise ValueError(f"This exporter assumes GS backend ('gs') for rgb. Got: {vision_backend}")

    gs_assets_reso = str(train_args.get("gs_assets_reso", "224"))
    gs_camera_id = int(train_args.get("gs_camera_id", 0))
    gs_camera_ids = train_args.get("gs_camera_ids", None)
    gs_camera_names = train_args.get("gs_camera_names", None)
    if args.gs_camera_ids:
        gs_camera_ids = [int(x) for x in args.gs_camera_ids.split(",") if x.strip()]
        gs_camera_names = None
    if args.gs_camera_names:
        gs_camera_names = [x.strip() for x in args.gs_camera_names.split(",") if x.strip()]
        gs_camera_ids = None
    gs_disable_bg = bool(train_args.get("gs_disable_bg", False))
    gs_minibatch = int(train_args.get("gs_minibatch", 32))
    gs_height = int(train_args.get("gs_height", 128))
    gs_width = int(train_args.get("gs_width", 128))
    bg_ply = train_args.get("gs_background_ply", None)

    episode_length = int(episode_seconds / max(ctrl_dt, 1e-9))

    # Build GS body map
    if train_args.get("gs_body_gaussians_json"):
        body_map = _load_body_map_from_json(train_args["gs_body_gaussians_json"])
    else:
        body_map = _default_airbot_body_map(gs_assets_reso)
    _validate_plys(body_map, bg_ply)

    # Build env (num_envs=1) to get exact obs/state dims
    config_overrides = {
        "action_repeat": int(action_repeat),
        "ctrl_dt": float(ctrl_dt),
        "episode_length": int(episode_length),
    }
    raw_env = registry.load(env_name, config_overrides=config_overrides)

    envs = MJXManiLikeVectorEnv(
        raw_env=raw_env,
        num_envs=1,
        seed=seed,
        episode_length=episode_length,
        action_repeat=action_repeat,
        vision=False,           # GS backend: env vision must be off
        num_vision_envs=1,
        randomization_fn=None,
        device_rank=(None if args.export_on_cpu else args.device_rank),
        auto_reset=True,
        include_state=include_state,
        debug_print_obs=False,

        vision_backend="gs",
        gs_body_gaussians=body_map,
        gs_background_ply=bg_ply,
        gs_camera_id=gs_camera_id,
        gs_camera_ids=gs_camera_ids,
        gs_camera_names=gs_camera_names,
        gs_height=gs_height,
        gs_width=gs_width,
        gs_minibatch=gs_minibatch,
        gs_disable_bg=gs_disable_bg,
    )

    obs, _ = envs.reset(seed=seed)
    # For export, keep inputs on the same device as the model
    obs = {k: v.to(device) for k, v in obs.items()}
    action_dim = int(envs.single_action_dim)

    # Decide whether to include state input in ONNX signature
    ckpt_has_state = _ckpt_has_state_branch(model_state)
    if args.force_with_state and args.force_no_state:
        raise ValueError("Cannot set both --force-with-state and --force-no-state")

    if args.force_with_state:
        export_with_state = True
    elif args.force_no_state:
        export_with_state = False
    else:
        # Auto: export with state if ckpt contains state branch
        export_with_state = ckpt_has_state

    # If exporting no-state but ckpt has state branch, this will not load cleanly
    if (not export_with_state) and ckpt_has_state:
        raise ValueError(
            "Checkpoint contains state branch (feat.state_fc...). "
            "You cannot export rgb-only model unless training was include_state=False "
            "or you change the architecture accordingly."
        )

    # If exporting with state but env obs doesn't include state, also inconsistent
    if export_with_state and ("state" not in obs):
        raise ValueError(
            "Requested export with state input but env.reset() obs has no 'state'. "
            "Check include_state and env obs structure."
        )

    agent = Agent(action_dim, obs).to(device)
    agent.load_state_dict(model_state, strict=True)
    agent.eval()

    if args.export_on_cpu:
        agent = agent.to("cpu")
        agent.eval()
        device = torch.device("cpu")
        obs = {k: v.to(device) for k, v in obs.items()}

    # Build wrapper and dummy inputs
    if export_with_state:
        policy = PolicyRGBState(agent).to(device).eval()
        dummy_rgb = obs["rgb"]
        dummy_state = obs["state"]
        input_names = ["rgb", "state"]
        output_names = ["action"]
        dynamic_axes = {
            "rgb": {0: "batch"},
            "state": {0: "batch"},
            "action": {0: "batch"},
        }
        dummy_inputs = (dummy_rgb, dummy_state)
    else:
        policy = PolicyRGB(agent).to(device).eval()
        dummy_rgb = obs["rgb"]
        input_names = ["rgb"]
        output_names = ["action"]
        dynamic_axes = {
            "rgb": {0: "batch"},
            "action": {0: "batch"},
        }
        dummy_inputs = (dummy_rgb,)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[export] ckpt={args.ckpt}")
    if gs_camera_names:
        cam_info = f"cam_names={gs_camera_names}"
    elif gs_camera_ids:
        cam_info = f"cam_ids={gs_camera_ids}"
    else:
        cam_info = f"cam_id={gs_camera_id}"
    print(f"[export] env={env_name} gs={gs_width}x{gs_height} {cam_info}")
    print(f"[export] signature={'(rgb,state)->action' if export_with_state else 'rgb->action'}")
    print(f"[export] output={args.output} device={device}")

    torch.onnx.export(
        policy,
        dummy_inputs,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print("[export] done.")


if __name__ == "__main__":
    main()
