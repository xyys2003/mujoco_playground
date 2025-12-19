from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

import json
import time
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio
import tyro
from torch.distributions.normal import Normal

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_mani_wrapper import MJXManiLikeVectorEnv


# ----------------------------
# Model definition (must match training)
# ----------------------------
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

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        z = self.feat(obs)
        mean = self.actor_mean(z)
        if deterministic:
            return mean
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        return dist.sample()


# ----------------------------
# Utilities
# ----------------------------
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
        "left": (assets / r / "left.ply").as_posix(),
        "right": (assets / r / "right.ply").as_posix(),
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


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3:
        raise ValueError(f"Unexpected frame ndim: {a.ndim}, shape={a.shape}")
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.shape[-1] != 3:
        raise ValueError(f"Unexpected channels: {a.shape[-1]} (shape={a.shape})")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _encode_mp4_ffmpeg(png_pattern: str, out_mp4: str, fps: int, pad16: bool = True) -> List[str]:
    # Use padding to 16-multiple for broad compatibility
    vf = "pad=ceil(iw/16)*16:ceil(ih/16)*16" if pad16 else "null"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", png_pattern,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_mp4,
    ]
    return cmd


# ----------------------------
# Eval script
# ----------------------------
@dataclass
class EvalArgs:
    ckpt: str
    out_dir: Optional[str] = None

    episodes: int = 1
    max_steps: int = 400
    deterministic: bool = True

    # dump frames
    save_every: int = 1

    # optional mp4 encoding (done at end of each episode)
    encode_mp4: bool = False
    fps: int = 25
    pad16: bool = True  # pad to multiple of 16 for compatibility

    # override render size (recommended 16-multiples to avoid codec issues)
    gs_height: Optional[int] = 64
    gs_width: Optional[int] = 96


def main():
    args = tyro.cli(EvalArgs)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get("args", {})
    env_name = str(train_args.get("env_name", "AirbotPlayPickCube"))

    # Pull env timing from ckpt args (fallback to reasonable defaults)
    episode_seconds = float(train_args.get("episode_seconds", 3.0))
    ctrl_dt = float(train_args.get("ctrl_dt", 0.04))
    action_repeat = int(train_args.get("action_repeat", 1))
    include_state = bool(train_args.get("include_state", True))
    seed = int(train_args.get("seed", 1))

    episode_length = int(episode_seconds / max(ctrl_dt, 1e-9))

    vision_backend = str(train_args.get("vision_backend", "gs")).lower().strip()
    if vision_backend != "gs":
        raise ValueError(f"This eval script is written for vision_backend='gs'. Got: {vision_backend}")

    gs_assets_reso = str(train_args.get("gs_assets_reso", "224"))
    gs_camera_id = int(train_args.get("gs_camera_id", 0))
    gs_disable_bg = bool(train_args.get("gs_disable_bg", False))
    gs_minibatch = int(train_args.get("gs_minibatch", 32))

    # allow override render size
    gs_height = int(args.gs_height if args.gs_height is not None else int(train_args.get("gs_height", 64)))
    gs_width = int(args.gs_width if args.gs_width is not None else int(train_args.get("gs_width", 96)))

    # output dir
    if args.out_dir is None:
        base = os.path.dirname(os.path.abspath(args.ckpt))
        args.out_dir = os.path.join(base, f"eval_{os.path.splitext(os.path.basename(args.ckpt))[0]}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Build GS body map
    if train_args.get("gs_body_gaussians_json"):
        body_map = _load_body_map_from_json(train_args["gs_body_gaussians_json"])
    else:
        body_map = _default_airbot_body_map(gs_assets_reso)
    bg_ply = train_args.get("gs_background_ply", None)
    _validate_plys(body_map, bg_ply)

    # Load env (IMPORTANT: do not set vision keys in config_overrides)
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
        vision=False,           # GS backend: must keep env vision off
        num_vision_envs=1,
        randomization_fn=None,
        device_rank=0,
        auto_reset=False,
        include_state=include_state,
        debug_print_obs=False,

        vision_backend="gs",
        gs_body_gaussians=body_map,
        gs_background_ply=bg_ply,
        gs_camera_id=gs_camera_id,
        gs_height=gs_height,
        gs_width=gs_width,
        gs_minibatch=gs_minibatch,
        gs_disable_bg=gs_disable_bg,
    )

    # Reset once to get obs shapes
    obs, _ = envs.reset(seed=seed)
    obs = {k: v.to(device) for k, v in obs.items()}
    action_dim = int(envs.single_action_dim)

    agent = Agent(action_dim, obs).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] out_dir={args.out_dir}")
    print(f"[eval] env={env_name}  episode_length={episode_length}  ctrl_dt={ctrl_dt}  action_repeat={action_repeat}")
    print(f"[eval] gs={gs_width}x{gs_height} camera_id={gs_camera_id} reso={gs_assets_reso} include_state={include_state}")
    print(f"[eval] episodes={args.episodes} max_steps={args.max_steps} deterministic={args.deterministic}")

    for ep in range(args.episodes):
        ep_dir = os.path.join(args.out_dir, f"episode_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        obs, _ = envs.reset(seed=seed + ep)
        obs = {k: v.to(device) for k, v in obs.items()}

        done = False
        frame_idx = 0
        ret = 0.0

        for t in range(args.max_steps):
            # save frame
            if (t % max(1, int(args.save_every))) == 0:
                if "rgb" in obs:
                    frame_t = obs["rgb"][0].detach()
                    if frame_t.is_floating_point():
                        frame_t = (frame_t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
                    else:
                        frame_t = frame_t.to(torch.uint8)
                    frame = _to_uint8_rgb(frame_t.cpu().numpy())
                    imageio.imwrite(os.path.join(ep_dir, f"frame_{frame_idx:06d}.png"), frame)
                    frame_idx += 1

            with torch.no_grad():
                action = agent.act(obs, deterministic=bool(args.deterministic))

            nobs, rew, term, trunc, info = envs.step(action.detach().cpu())
            # env returns CPU tensors; move back
            obs = {k: v.to(device) for k, v in nobs.items()}
            r = float(rew.item()) if hasattr(rew, "item") else float(rew)
            ret += r
            done = bool((term | trunc).item())
            if done:
                break

        print(f"[eval] episode {ep:03d}: steps={t+1} saved_frames={frame_idx} return={ret:.3f} done={done}")

        # Optional: encode mp4
        if args.encode_mp4:
            png_pattern = os.path.join(ep_dir, "frame_%06d.png")
            out_mp4 = os.path.join(args.out_dir, f"episode_{ep:03d}.mp4")
            cmd = _encode_mp4_ffmpeg(png_pattern, out_mp4, fps=int(args.fps), pad16=bool(args.pad16))
            print("[eval] ffmpeg:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print(f"[eval] wrote {out_mp4}")
            except Exception as exc:
                print(f"[eval] ffmpeg failed: {exc}")
                print(f"[eval] You can run manually:\n  {' '.join(cmd)}")

    print("[eval] done.")


if __name__ == "__main__":
    main()
