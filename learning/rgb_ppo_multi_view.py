from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

import json
import time
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from mujoco_playground import dm_control_suite,registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_mani_wrapper import MJXManiLikeVectorEnv


@dataclass
class Args:
    # env
    env_name: str = "AirbotPlayPickCube"
    num_envs: int = 256
    seed: int = 1
    episode_seconds: float = 3.0
    ctrl_dt: float = 0.04
    action_repeat: int = 1
    include_state: bool = True
    device_rank: int = 0

    # NEW: vision backend
    vision_backend: str = "gs"  # "madrona" | "gs"

    # NEW: GS config
    gs_assets_reso: str = "224"
    gs_height: int = 60
    gs_width: int = 80
    gs_minibatch: int = 32
    gs_camera_id: int = 0
    gs_camera_ids: Optional[List[int]] = None
    gs_camera_names: Optional[List[str]] = None
    gs_disable_bg: bool = False

    # optional: override mapping via json
    gs_body_gaussians_json: Optional[str] = None
    gs_background_ply: Optional[str] = None

    # PPO (keep minimal here)
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    num_steps: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 32
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    reward_scale: float = 1.0
    success_once: bool = True

    # logging
    exp_name: Optional[str] = None
    save_model: bool = True
    checkpoint_interval: int = 10  # in PPO updates
    resume_from: Optional[str] = None
    eval_episodes: int = 0  # >0 to run evaluation after training (or eval_only)
    eval_deterministic: bool = True
    eval_only: bool = False

    # debug rendering
    debug_render_epochs: int = 3  # save rollout renderings for first N PPO updates
    debug_render_dir: Optional[str] = None


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """
    CNN encoder for RGB observations shaped [B, H, W, C] (uint8 or float).
    Optionally concatenates a low-dimensional 'state' vector if present.

    Fix: infer CNN flat dim using a CPU dummy tensor so init does NOT depend on
    sample_obs device (CPU/GPU), avoiding dtype/device mismatch.
    """

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

        # Infer flat dim using CPU dummy (weights are on CPU during __init__)
        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, int(H), int(W)), dtype=torch.float32)  # CPU
            n_flat = int(self.cnn(dummy).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(),
        )

        self.state_fc = None
        self.out_dim = 256
        if "state" in sample_obs:
            s = sample_obs["state"]
            if s.ndim != 2:
                raise ValueError(f"Expected state obs shape [B,D], got {tuple(s.shape)}")
            sdim = int(s.shape[-1])
            self.state_fc = nn.Sequential(
                nn.Linear(sdim, 256),
                nn.ReLU(),
            )
            self.out_dim += 256

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs["rgb"].float().permute(0, 3, 1, 2) / 255.0  # [B,C,H,W]
        z = self.fc(self.cnn(x))
        if self.state_fc is not None and ("state" in obs):
            z2 = self.state_fc(obs["state"].float())
            z = torch.cat([z, z2], dim=1)
        return z


class Agent(nn.Module):
    def __init__(self, action_dim: int, sample_obs: Dict[str, torch.Tensor]):
        super().__init__()
        self.feat = NatureCNN(sample_obs)
        self.critic = nn.Sequential(layer_init(nn.Linear(self.feat.out_dim, 512)), nn.ReLU(), layer_init(nn.Linear(512, 1)))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(self.feat.out_dim, 512)), nn.ReLU(), layer_init(nn.Linear(512, action_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_action_and_value(self, obs: Dict[str, torch.Tensor], action: Optional[torch.Tensor] = None):
        z = self.feat(obs)
        mean = self.actor_mean(z)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        ent = dist.entropy().sum(1)
        value = self.critic(z).view(-1)
        return action, logprob, ent, value


def _load_body_map_from_json(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        m = json.load(f)
    if not isinstance(m, dict):
        raise ValueError("gs_body_gaussians_json must be a JSON dict: {body_name: ply_path}")
    return {str(k): str(v) for k, v in m.items()}


def _default_airbot_body_map(reso: str) -> Dict[str, str]:
    # exactly your notebook pattern
    ASSETS_PATH = mjx_env.ROOT_PATH / "manipulation/airbot_play/3dgs"
    Reso = str(reso)
    body_gaussians = {
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
    return body_gaussians


def _validate_plys(body_map: Dict[str, str], background_ply: Optional[str]):
    missing = []
    for k, p in body_map.items():
        if not os.path.exists(p):
            missing.append(f"{k}: {p}")
    if background_ply is not None and not os.path.exists(background_ply):
        missing.append(f"background: {background_ply}")
    if missing:
        msg = "Missing PLY files:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)


def _save_checkpoint(
    args: Args,
    run_name: str,
    agent: Agent,
    optimizer: optim.Optimizer,
    global_step: int,
    update: int,
    path: Optional[str] = None,
):
    ckpt_dir = path or f"runs/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{update:05d}.pt")
    torch.save(
        {
            "args": asdict(args),
            "run_name": run_name,
            "model_state": agent.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "global_step": global_step,
            "update": update,
        },
        ckpt_path,
    )
    print(f"[checkpoint] saved to {ckpt_path}")


def _save_debug_video(
    frames: List[np.ndarray],
    path: str,
    fps: int,
    first_frame_path: Optional[str] = None,
) -> None:
    """Save a short rollout video for debugging (no-op if frames empty)."""

    if not frames:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    proc: List[np.ndarray] = []
    for f in frames:
        a = np.asarray(f)

        # Ensure HWC
        if a.ndim == 2:  # gray -> RGB
            a = np.repeat(a[..., None], 3, axis=2)
        if a.ndim != 3:
            continue

        # Ensure 3 channels
        if a.shape[-1] == 4:  # RGBA -> RGB
            a = a[..., :3]
        if a.shape[-1] != 3:
            continue

        # Ensure uint8 [0,255]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)

        proc.append(a)

    if not proc:
        return

    # Save first frame png if requested
    if first_frame_path is not None:
        try:
            imageio.imwrite(first_frame_path, proc[0])
            print(f"[debug] saved rollout first frame to {first_frame_path}")
        except Exception as exc:
            print(f"[debug] failed to save first frame to {first_frame_path}: {exc}")

    # Ensure consistent shape across frames (drop inconsistent frames)
    H, W, C = proc[0].shape
    proc = [p for p in proc if p.shape == (H, W, C)]
    if not proc:
        return

    # Write mp4 robustly (avoid macroblock auto-resize + enforce pix_fmt)
    try:
        writer = imageio.get_writer(
            path,
            fps=max(1, int(fps)),
            format="ffmpeg",
            codec="libx264",
            macro_block_size=16,  
        )
        try:
            for p in proc:
                writer.append_data(p)
        finally:
            writer.close()
        print(f"[debug] saved rollout render to {path}")
    except Exception as exc:
        print(f"[debug] failed to save rollout render to {path}: {exc}")


def _save_frame_sequence(
    frames: List[np.ndarray],
    out_dir: str,
    prefix: str = "frame",
) -> None:
    if not frames:
        return
    os.makedirs(out_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}_{idx:04d}.png")
        try:
            imageio.imwrite(path, frame)
        except Exception as exc:
            print(f"[debug] failed to save frame {path}: {exc}")


@torch.no_grad()
def _evaluate_agent(
    agent: Agent,
    envs: MJXManiLikeVectorEnv,
    device: torch.device,
    num_episodes: int,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
):
    agent.eval()

    obs, _ = envs.reset()
    for k, v in obs.items():
        if v.device != device:
            obs[k] = v.to(device)

    ep_returns = torch.zeros(envs.num_envs, device=device)
    completed = []
    steps = 0
    max_allowed_steps = max_steps or (num_episodes * 10_000)

    while len(completed) < num_episodes and steps < max_allowed_steps:
        if deterministic:
            z = agent.feat(obs)
            mean = agent.actor_mean(z)
            action = mean
        else:
            action, _, _, _ = agent.get_action_and_value(obs)

        env_action = action.detach().to(device=device, dtype=torch.float32).contiguous()
        nobs, rew, term, trunc, _ = envs.step(env_action)
        for k, v in nobs.items():
            if v.device != device:
                nobs[k] = v.to(device)
        if rew.device != device:
            rew = rew.to(device)
        if term.device != device:
            term = term.to(device)
        if trunc.device != device:
            trunc = trunc.to(device)
        done = term | trunc

        ep_returns += rew
        if done.any():
            done_idx = done.nonzero(as_tuple=False).view(-1)
            for idx in done_idx:
                completed.append(float(ep_returns[idx]))
                if len(completed) >= num_episodes:
                    break
            ep_returns[done_idx] = 0.0
        obs = nobs
        steps += 1

    return completed


def _split_multi_view_frames(rgb: torch.Tensor) -> List[np.ndarray]:
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
    num_views = frame.shape[-1] // 3
    views = frame.reshape(frame.shape[0], frame.shape[1], num_views, 3)
    views = np.transpose(views, (2, 0, 1, 3))
    return list(views)


def main():
    args = tyro.cli(Args)

    run_name = args.exp_name or f"rgbppo_multi_view_{args.env_name}__{args.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")

    debug_render_dir = args.debug_render_dir or os.path.join("runs", run_name, "render_debug")
    debug_render_epochs = max(0, int(args.debug_render_epochs))
    debug_render_fps = int(round(1.0 / max(args.ctrl_dt * args.action_repeat, 1e-6)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_length = int(args.episode_seconds / args.ctrl_dt)

    vb = (args.vision_backend or "madrona").lower().strip()
    env_vision = (vb == "madrona")  # GS backend must keep env vision off

    config_overrides = {
        # "vision": bool(env_vision),
        # "vision_config.render_batch_size": int(args.num_envs),
        "action_repeat": int(args.action_repeat),
        "ctrl_dt": float(args.ctrl_dt),
        "episode_length": int(episode_length),
    }

    raw_env = registry.load(args.env_name, config_overrides=config_overrides)

    # GS mapping
    gs_body_map: Dict[str, str] = {}
    if vb == "gs":
        if args.gs_body_gaussians_json is not None:
            gs_body_map = _load_body_map_from_json(args.gs_body_gaussians_json)
        else:
            gs_body_map = _default_airbot_body_map(args.gs_assets_reso)
        _validate_plys(gs_body_map, args.gs_background_ply)

    envs = MJXManiLikeVectorEnv(
        raw_env=raw_env,
        num_envs=args.num_envs,
        seed=args.seed,
        episode_length=episode_length,
        action_repeat=args.action_repeat,
        vision=env_vision,
        num_vision_envs=args.num_envs,
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

    obs, _ = envs.reset(seed=args.seed)
    for k, v in obs.items():
        if v.device != device:
            obs[k] = v.to(device)
    if args.include_state and "state" not in obs:
        raise RuntimeError("include_state=True but env obs has no key 'state'")

    action_dim = int(envs.single_action_dim)
    agent = Agent(action_dim, obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # Optional resume
    start_update = 1
    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        agent.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        global_step = int(ckpt.get("global_step", 0))
        start_update = int(ckpt.get("update", 0)) + 1
        prev_run = ckpt.get("run_name")
        if isinstance(prev_run, str):
            print(f"[resume] previous run name: {prev_run}")
        print(f"[resume] start update {start_update}, global_step {global_step}")

    # rollout buffers (minimal)
    num_steps = args.num_steps
    num_envs = args.num_envs
    obs_buf = {k: torch.zeros((num_steps, num_envs) + tuple(v.shape[1:]), dtype=v.dtype, device=device) for k, v in obs.items()}
    act_buf = torch.zeros((num_steps, num_envs, action_dim), device=device)
    logp_buf = torch.zeros((num_steps, num_envs), device=device)
    rew_buf = torch.zeros((num_steps, num_envs), device=device)
    term_buf = torch.zeros((num_steps, num_envs), device=device, dtype=torch.bool)
    trunc_buf = torch.zeros((num_steps, num_envs), device=device, dtype=torch.bool)
    val_buf = torch.zeros((num_steps, num_envs), device=device)

    next_obs = obs
    batch_size = num_steps * num_envs
    if batch_size % args.num_minibatches != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by num_minibatches ({args.num_minibatches})"
        )
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    if args.eval_only and args.eval_episodes <= 0:
        raise ValueError("eval_only=True requires eval_episodes > 0")

    if not args.eval_only:
        for update in range(start_update, num_updates + 1):
            update_t0 = time.time()
            debug_frames: List[np.ndarray] = []
            debug_frames_by_view: Optional[List[List[np.ndarray]]] = None
            record_debug = (debug_render_epochs > 0) and ((update - start_update) < debug_render_epochs)
            episode_frames: List[np.ndarray] = []
            episode_complete = False
            rollout_logs: Dict[str, List[float]] = defaultdict(list)
            success_once_tracker = torch.zeros(num_envs, device=device, dtype=torch.bool)

            for step in range(num_steps):
                global_step += num_envs
                for k in obs_buf:
                    obs_buf[k][step] = next_obs[k]

                with torch.no_grad():
                    a, lp, ent, v = agent.get_action_and_value(next_obs)
                act_buf[step] = a
                logp_buf[step] = lp
                val_buf[step] = v

                env_action = a.detach().to(device=device, dtype=torch.float32).contiguous()
                nobs, rew, term, trunc, infos = envs.step(env_action)
                for k, v in nobs.items():
                    if v.device != device:
                        nobs[k] = v.to(device)
                if rew.device != device:
                    rew = rew.to(device)
                if term.device != device:
                    term = term.to(device)
                if trunc.device != device:
                    trunc = trunc.to(device)
                rew = rew * float(args.reward_scale)
                done = term | trunc

                rew_buf[step] = rew
                term_buf[step] = term
                trunc_buf[step] = trunc
                next_obs = nobs

                success_values = None
                for k, v in infos.get("log", {}).items():
                    if isinstance(v, torch.Tensor):
                        rollout_logs[k].append(v.float().mean().item())
                    else:
                        try:
                            rollout_logs[k].append(float(v))
                        except Exception:
                            continue

                    if k in ("success", "is_success"):
                        if isinstance(v, torch.Tensor) and v.numel() == num_envs:
                            success_values = v.to(device).view(-1)
                        else:
                            try:
                                value = torch.as_tensor(v, device=device)
                                if value.numel() == num_envs:
                                    success_values = value.view(-1)
                            except Exception:
                                success_values = None
                    elif k == "reward/success":
                        if isinstance(v, torch.Tensor) and v.numel() == num_envs:
                            success_values = (v > 0).to(device).view(-1)
                        else:
                            try:
                                value = torch.as_tensor(v, device=device)
                                if value.numel() == num_envs:
                                    success_values = value.view(-1) > 0
                            except Exception:
                                success_values = None

                if args.success_once:
                    if success_values is not None:
                        success_once_tracker |= success_values.bool().view(-1)
                    if done.any():
                        ended = done
                        rollout_logs["success_once"].append(
                            success_once_tracker[ended].float().mean().item()
                        )
                        success_once_tracker[ended] = False

                if record_debug:
                    frame_t = next_obs.get("rgb")
                    if frame_t is not None:
                        view_frames = _split_multi_view_frames(frame_t[0])
                        if view_frames:
                            if debug_frames_by_view is None:
                                debug_frames_by_view = [[] for _ in range(len(view_frames))]
                            for idx, view_frame in enumerate(view_frames):
                                if idx < len(debug_frames_by_view):
                                    debug_frames_by_view[idx].append(view_frame)
                            composite = (
                                np.concatenate(view_frames, axis=1)
                                if len(view_frames) > 1
                                else view_frames[0]
                            )
                            debug_frames.append(composite)

                if not episode_complete:
                    frame_t = next_obs.get("rgb")
                    if frame_t is not None:
                        view_frames = _split_multi_view_frames(frame_t[0])
                        if view_frames:
                            composite = (
                                np.concatenate(view_frames, axis=1)
                                if len(view_frames) > 1
                                else view_frames[0]
                            )
                            episode_frames.append(composite)
                    if done[0].item():
                        episode_complete = True

            rollout_reward_mean = float(rew_buf.mean().item())
            rollout_return_mean = float(rew_buf.sum(dim=0).mean().item())
            done_rate = float((term_buf | trunc_buf).float().mean().item())
            termination_rate = float(term_buf.float().mean().item())
            truncation_rate = float(trunc_buf.float().mean().item())
            action_abs_mean = float(act_buf.abs().mean().item())
            action_clip_frac = float((act_buf.abs() > 0.99).float().mean().item())

            # GAE
            with torch.no_grad():
                _, _, _, next_value = agent.get_action_and_value(next_obs, action=None)
                adv = torch.zeros_like(rew_buf, device=device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - term_buf[t].float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - term_buf[t].float()
                        nextvalues = val_buf[t + 1]
                    delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    adv[t] = lastgaelam
                ret = adv + val_buf

            # flatten
            b_obs = {k: v.reshape((-1,) + v.shape[2:]) for k, v in obs_buf.items()}
            b_act = act_buf.reshape((-1, action_dim))
            b_logp = logp_buf.reshape(-1)
            b_adv = adv.reshape(-1)
            b_ret = ret.reshape(-1)
            b_val = val_buf.reshape(-1)

            # PPO diagnostics
            y_pred = b_val.detach().cpu().numpy()
            y_true = b_ret.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_variance = np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

            approx_kl_vals = []
            clipfrac_vals = []
            ratio_vals = []

            inds = np.arange(batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = inds[start:start + minibatch_size]
                    mb_obs = {k: v[mb] for k, v in b_obs.items()}

                    new_a, new_logp, ent, new_v = agent.get_action_and_value(mb_obs, b_act[mb])
                    ratio = (new_logp - b_logp[mb]).exp()
                    with torch.no_grad():
                        approx_kl_vals.append((b_logp[mb] - new_logp).mean().item())
                        clipfrac_vals.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                        ratio_vals.append(ratio.mean().item())

                    mb_adv = b_adv[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_loss = 0.5 * (new_v - b_ret[mb]).pow(2).mean()
                    ent_loss = ent.mean()
                    loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            writer.add_scalar("loss/pg", float(pg_loss.item()), global_step)
            writer.add_scalar("loss/v", float(v_loss.item()), global_step)
            writer.add_scalar("loss/entropy", float(ent_loss.item()), global_step)
            if approx_kl_vals:
                writer.add_scalar("charts/approx_kl", float(np.mean(approx_kl_vals)), global_step)
            if clipfrac_vals:
                writer.add_scalar("charts/clipfrac", float(np.mean(clipfrac_vals)), global_step)
            if ratio_vals:
                ratio_mean = float(np.mean(ratio_vals))
                ratio_std = float(np.std(ratio_vals))
                writer.add_scalar("charts/ratio_mean", ratio_mean, global_step)
                writer.add_scalar("charts/ratio_std", ratio_std, global_step)
            update_dt = max(1e-6, time.time() - update_t0)
            fps = float((num_envs * num_steps) / update_dt)
            writer.add_scalar("perf/fps", fps, global_step)
            writer.add_scalar("rollout/reward_mean", rollout_reward_mean, global_step)
            writer.add_scalar("rollout/return_mean", rollout_return_mean, global_step)
            writer.add_scalar("env/done_rate", done_rate, global_step)
            writer.add_scalar("env/termination_rate", termination_rate, global_step)
            writer.add_scalar("env/truncation_rate", truncation_rate, global_step)
            writer.add_scalar("policy/logstd_mean", float(agent.actor_logstd.mean().item()), global_step)
            writer.add_scalar("policy/logstd_max", float(agent.actor_logstd.max().item()), global_step)
            writer.add_scalar("policy/logstd_min", float(agent.actor_logstd.min().item()), global_step)
            writer.add_scalar("policy/action_abs_mean", action_abs_mean, global_step)
            writer.add_scalar("policy/action_clip_frac", action_clip_frac, global_step)
            if not np.isnan(explained_variance):
                writer.add_scalar("charts/explained_variance", explained_variance, global_step)
            writer.add_scalar("value/vpred_mean", float(b_val.mean().item()), global_step)
            writer.add_scalar("value/return_mean", float(b_ret.mean().item()), global_step)
            writer.add_scalar("value/return_std", float(b_ret.std().item()), global_step)
            writer.add_scalar("charts/step", global_step, global_step)
            for log_key, values in rollout_logs.items():
                if values:
                    writer.add_scalar(f"env/{log_key}", float(np.mean(values)), global_step)

            if record_debug and debug_frames:
                vid_path = os.path.join(debug_render_dir, f"update_{update:03d}.mp4")
                frame_path = os.path.join(debug_render_dir, f"update_{update:03d}_first.png")
                writer.add_image(
                    "debug/first_frame",
                    debug_frames[0],
                    global_step=global_step,
                    dataformats="HWC",
                )
                _save_debug_video(
                    debug_frames,
                    vid_path,
                    fps=debug_render_fps,
                    first_frame_path=frame_path,
                )
                if debug_frames_by_view:
                    for idx, view_frames in enumerate(debug_frames_by_view):
                        if not view_frames:
                            continue
                        view_vid_path = os.path.join(
                            debug_render_dir, f"update_{update:03d}_view_{idx:02d}.mp4"
                        )
                        view_frame_path = os.path.join(
                            debug_render_dir, f"update_{update:03d}_view_{idx:02d}_first.png"
                        )
                        _save_debug_video(
                            view_frames,
                            view_vid_path,
                            fps=debug_render_fps,
                            first_frame_path=view_frame_path,
                        )

            if args.save_model and update % max(1, args.checkpoint_interval) == 0:
                _save_checkpoint(
                    args=args,
                    run_name=run_name,
                    agent=agent,
                    optimizer=optimizer,
                    global_step=global_step,
                    update=update,
                )
                if episode_frames:
                    frame_dir = os.path.join(
                        "runs", run_name, "ckpt_renders", f"update_{update:05d}"
                    )
                    _save_frame_sequence(episode_frames, frame_dir, prefix="episode")

    if args.save_model and not args.eval_only:
        final_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), final_path)
        print(f"[checkpoint] final model saved to {final_path}")

    if args.eval_episodes > 0:
        eval_max_steps = args.eval_episodes * (episode_length * args.action_repeat + 10)
        returns = _evaluate_agent(
            agent=agent,
            envs=envs,
            device=device,
            num_episodes=args.eval_episodes,
            deterministic=args.eval_deterministic,
            max_steps=eval_max_steps,
        )
        if returns:
            mean_ret = float(np.mean(returns))
            print(f"[eval] mean reward over {len(returns)} episodes: {mean_ret:.3f}")
            writer.add_scalar("eval/episode_reward", mean_ret, global_step)
        else:
            print("[eval] no completed episodes during evaluation")

    writer.close()


if __name__ == "__main__":
    main()
