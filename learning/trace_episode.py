# trace_episode.py
from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_mani_wrapper import MJXManiLikeVectorEnv


# ----------------------------
# Model (copy from your train script)
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

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        z = self.feat(obs)
        mean = self.actor_mean(z)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = mean if deterministic else dist.sample()
        logprob = dist.log_prob(action).sum(1)
        ent = dist.entropy().sum(1)
        value = self.critic(z).view(-1)
        return action, logprob, ent, value


# ----------------------------
# GS helpers (copy from your train script)
# ----------------------------
def _load_body_map_from_json(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("gs_body_gaussians_json must be a JSON dict: {body_name: ply_path}")
    return {str(k): str(v) for k, v in m.items()}


def _default_airbot_body_map(reso: str) -> Dict[str, str]:
    ASSETS_PATH = mjx_env.ROOT_PATH / "manipulation/airbot_play/3dgs"
    Reso = str(reso)
    return {
        "arm_base": (ASSETS_PATH / Reso / "arm_base.ply").as_posix(),
        "link1": (ASSETS_PATH / Reso / "link1.ply").as_posix(),
        "link2": (ASSETS_PATH / Reso / "link2.ply").as_posix(),
        "link3": (ASSETS_PATH / Reso / "link3.ply").as_posix(),
        "link4": (ASSETS_PATH / Reso / "link4.ply").as_posix(),
        "link5": (ASSETS_PATH / Reso / "link5.ply").as_posix(),
        "link6": (ASSETS_PATH / Reso / "link6.ply").as_posix(),
        "left": (ASSETS_PATH / Reso / "left.ply").as_posix(),
        "right": (ASSETS_PATH / Reso / "right.ply").as_posix(),
        "box": (ASSETS_PATH / "green_cube.ply").as_posix(),
    }


def _validate_plys(body_map: Dict[str, str], background_ply: Optional[str]):
    missing = []
    for k, p in body_map.items():
        if not os.path.exists(p):
            missing.append(f"{k}: {p}")
    if background_ply is not None and not os.path.exists(background_ply):
        missing.append(f"background: {background_ply}")
    if missing:
        raise FileNotFoundError("Missing PLY files:\n" + "\n".join(missing))


# ----------------------------
# Video / frame utils
# ----------------------------
def _split_multi_view_frames(rgb: torch.Tensor) -> List[np.ndarray]:
    """
    rgb: [H,W,C] torch tensor (uint8 or float)
    If C == 3 -> single view
    If C == 3*n -> split into n views
    """
    if rgb.ndim != 3:
        return []
    if rgb.is_floating_point():
        rgb = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    else:
        rgb = rgb.to(torch.uint8)
    frame = rgb.detach().cpu().numpy()
    if frame.shape[-1] == 3:
        return [frame]
    if frame.shape[-1] % 3 != 0:
        return [frame[..., :3]]
    n = frame.shape[-1] // 3
    views = frame.reshape(frame.shape[0], frame.shape[1], n, 3)
    views = np.transpose(views, (2, 0, 1, 3))  # [n,H,W,3]
    return list(views)


def _pad_to_even_rgb_uint8(a: np.ndarray) -> np.ndarray:
    """Ensure a is HxWx3 uint8, contiguous, and has even H/W by padding."""
    a = np.asarray(a)

    # handle grayscale
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    # drop alpha if present
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]

    # if weird channel count, try best-effort: keep first 3
    if a.ndim != 3:
        raise ValueError(f"Bad frame ndim={a.ndim}")
    if a.shape[-1] != 3:
        a = a[..., :3]

    # dtype to uint8
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)

    H, W, C = a.shape
    if C != 3:
        raise ValueError(f"Bad channel count: {C}")

    # pad to even dims (safe for ffmpeg later; also harmless for png/jpg)
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        a2 = np.zeros((H + pad_h, W + pad_w, 3), dtype=np.uint8)
        a2[:H, :W, :] = a
        # edge pad to reduce black border artifacts
        if pad_h:
            a2[H:H + 1, :W, :] = a[H - 1:H, :W, :]
        if pad_w:
            a2[:H, W:W + 1, :] = a[:H, W - 1:W, :]
        if pad_h and pad_w:
            a2[H:H + 1, W:W + 1, :] = a[H - 1:H, W - 1:W, :]
        a = a2

    return np.ascontiguousarray(a)


class FrameSink:
    """Stream frames to disk as numbered images to avoid ffmpeg trailer errors."""
    def __init__(self, out_dir: str, ep_tag: str, ext: str = "png", every: int = 1):
        self.dir = Path(out_dir) / "frames" / ep_tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ext = ext.lower().lstrip(".")
        self.every = max(1, int(every))
        self.total_seen = 0
        self.saved = 0

    def add(self, frame: np.ndarray) -> None:
        self.total_seen += 1
        if (self.total_seen - 1) % self.every != 0:
            return
        try:
            a = _pad_to_even_rgb_uint8(frame)
        except Exception:
            return
        path = self.dir / f"frame_{self.saved:06d}.{self.ext}"
        imageio.imwrite(path.as_posix(), a)
        self.saved += 1


def _to_float_scalar(x: Any) -> Optional[float]:
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.item())
            # for vector-like values: take mean for logging
            return float(x.float().mean().item())
        return float(x)
    except Exception:
        return None


def _dump_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[trace] saved csv: {path}")


# ----------------------------
# Trace args
# ----------------------------
@dataclass
class TraceArgs:
    # checkpoint & output
    ckpt: str
    out_dir: str = "runs/trace_episode"
    tb_dir: Optional[str] = None

    # env config (match your train script)
    env_name: str = "AirbotPlayPickCube"
    seed: int = 1
    episode_seconds: float = 5.0
    ctrl_dt: float = 0.04
    action_repeat: int = 1
    include_state: bool = True
    device_rank: int = 0

    # vision backend
    vision_backend: str = "gs"  # "madrona" | "gs"
    gs_assets_reso: str = "224"
    gs_height: int = 128
    gs_width: int = 128
    gs_minibatch: int = 16
    gs_camera_id: int = 0
    gs_camera_ids: Optional[List[int]] = None
    gs_camera_names: Optional[List[str]] = None
    gs_disable_bg: bool = False
    gs_body_gaussians_json: Optional[str] = None
    gs_background_ply: Optional[str] = None

    # trace behavior
    num_episodes: int = 1
    deterministic: bool = True
    max_steps: Optional[int] = None  # default: episode_length * action_repeat + buffer

    # outputs
    save_video: bool = False  # keep but default off (avoid ffmpeg trailer / I/O errors)
    save_images: bool = True  # save numbered frames instead
    image_ext: str = "png"
    image_every: int = 1      # save every N frames (1 = all)
    save_csv: bool = True

    print_keys: bool = False  # print available log keys at first step


def _build_env(args: TraceArgs) -> Tuple[MJXManiLikeVectorEnv, torch.device, int]:
    episode_length = int(args.episode_seconds / args.ctrl_dt)

    vb = (args.vision_backend or "madrona").lower().strip()
    env_vision = (vb == "madrona")  # GS backend must keep env vision off

    config_overrides = {
        "action_repeat": int(args.action_repeat),
        "ctrl_dt": float(args.ctrl_dt),
        "episode_length": int(episode_length),
    }
    raw_env = registry.load(args.env_name, config_overrides=config_overrides)

    gs_body_map: Dict[str, str] = {}
    if vb == "gs":
        if args.gs_body_gaussians_json is not None:
            gs_body_map = _load_body_map_from_json(args.gs_body_gaussians_json)
        else:
            gs_body_map = _default_airbot_body_map(args.gs_assets_reso)
        _validate_plys(gs_body_map, args.gs_background_ply)

    envs = MJXManiLikeVectorEnv(
        raw_env=raw_env,
        num_envs=1,
        seed=args.seed,
        episode_length=episode_length,
        action_repeat=args.action_repeat,
        vision=env_vision,
        num_vision_envs=1,
        randomization_fn=None,
        device_rank=args.device_rank,
        auto_reset=True,
        include_state=args.include_state,
        debug_print_obs=False,

        vision_backend=vb,
        gs_body_gaussians=gs_body_map if vb == "gs" else None,
        gs_background_ply=args.gs_background_ply,
        gs_camera_id=args.gs_camera_id,
        gs_camera_ids=args.gs_camera_ids,
        gs_camera_names=args.gs_camera_names,
        gs_height=args.gs_height,
        gs_width=args.gs_width,
        gs_minibatch=int(args.gs_minibatch) if vb == "gs" else None,
        gs_disable_bg=args.gs_disable_bg,
    )

    device = envs.torch_device
    return envs, device, episode_length


def _load_agent(ckpt_path: str, sample_obs: Dict[str, torch.Tensor], action_dim: int, device: torch.device) -> Agent:
    agent = Agent(action_dim, sample_obs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        agent.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        agent.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unrecognized ckpt format: {type(ckpt)}")

    agent.eval()
    return agent


def main():
    args = tyro.cli(TraceArgs)
    os.makedirs(args.out_dir, exist_ok=True)
    tb_dir = args.tb_dir or os.path.join(args.out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    envs, device, episode_length = _build_env(args)

    obs, _ = envs.reset(seed=args.seed)
    for k, v in obs.items():
        if v.device != device:
            obs[k] = v.to(device)
    if args.include_state and "state" not in obs:
        raise RuntimeError("include_state=True but env obs has no key 'state'")

    action_dim = int(envs.single_action_dim)
    agent = _load_agent(args.ckpt, obs, action_dim, device)

    fps = int(round(1.0 / max(args.ctrl_dt * args.action_repeat, 1e-6)))
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(episode_length * args.action_repeat + 10)

    writer = SummaryWriter(tb_dir)
    print(f"[trace] tb_dir={tb_dir}")
    print(f"[trace] device={device}, fps={fps}, max_steps={max_steps}")

    global_step_base = 0

    for ep in range(args.num_episodes):
        obs, _ = envs.reset(seed=args.seed + ep)
        for k, v in obs.items():
            if v.device != device:
                obs[k] = v.to(device)

        ep_tag = f"ep{ep:02d}"
        sink = FrameSink(args.out_dir, ep_tag=ep_tag, ext=args.image_ext, every=args.image_every) if args.save_images else None
        rows: List[Dict[str, Any]] = []

        ep_return = 0.0
        ep_step = 0
        printed_keys = False

        while ep_step < max_steps:
            # record PRE-step frame
            rgb0 = obs.get("rgb")
            if rgb0 is not None:
                view_frames = _split_multi_view_frames(rgb0[0])
                if view_frames:
                    composite = np.concatenate(view_frames, axis=1) if len(view_frames) > 1 else view_frames[0]
                    if sink is not None:
                        sink.add(composite)

            with torch.no_grad():
                a, _, _, _ = agent.get_action_and_value(obs, deterministic=args.deterministic)

            nobs, rew, term, trunc, infos = envs.step(a.detach().to(device=device, dtype=torch.float32).contiguous())

            for k, v in nobs.items():
                if v.device != device:
                    nobs[k] = v.to(device)
            if rew.device != device:
                rew = rew.to(device)
            if term.device != device:
                term = term.to(device)
            if trunc.device != device:
                trunc = trunc.to(device)

            done = (term | trunc)
            r = float(rew[0].item())
            ep_return += r

            global_step = global_step_base + ep_step

            row: Dict[str, Any] = {
                "episode": ep,
                "t": ep_step,
                "global_t": global_step,
                "reward": r,
                "term": int(term[0].item()),
                "trunc": int(trunc[0].item()),
                "done": int(done[0].item()),
            }

            log_dict = infos.get("log", {}) if isinstance(infos, dict) else {}
            if isinstance(log_dict, dict):
                if args.print_keys and not printed_keys and log_dict:
                    print(f"[trace] available log keys: {sorted(list(log_dict.keys()))}")
                    printed_keys = True

                for k, v in log_dict.items():
                    fv = _to_float_scalar(v)
                    if fv is None:
                        continue
                    row[k] = fv
                    writer.add_scalar(f"trace/{k}", fv, global_step)

            writer.add_scalar("trace/reward", r, global_step)
            writer.add_scalar("trace/ep_return_running", ep_return, global_step)
            writer.add_scalar("trace/done", float(done[0].item()), global_step)
            writer.add_scalar("trace/trunc", float(trunc[0].item()), global_step)
            writer.add_scalar("trace/term", float(term[0].item()), global_step)

            rows.append(row)

            if bool(done[0].item()):
                # append FINAL frame using final_observation (if present)
                fin = infos.get("final_observation", None) if isinstance(infos, dict) else None
                if isinstance(fin, dict) and ("rgb" in fin):
                    frgb = fin["rgb"]
                    if isinstance(frgb, torch.Tensor):
                        if frgb.device != device:
                            frgb = frgb.to(device)
                        view_frames = _split_multi_view_frames(frgb[0])
                        if view_frames:
                            composite = np.concatenate(view_frames, axis=1) if len(view_frames) > 1 else view_frames[0]
                            if sink is not None:
                                sink.add(composite)
                break

            obs = nobs
            ep_step += 1

        writer.flush()

        if args.save_csv:
            csv_path = os.path.join(args.out_dir, f"{ep_tag}.csv")
            _dump_csv(rows, csv_path)

        if args.save_images and sink is not None:
            print(f"[trace] saved frames: {os.path.join(args.out_dir, 'frames', ep_tag)} (count={sink.saved})")

        print(f"[trace] episode {ep}: steps={len(rows)}, return={ep_return:.3f}")

        # next episode: keep TB steps monotonic
        global_step_base += max_steps

    writer.close()


if __name__ == "__main__":
    main()
