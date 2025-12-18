from __future__ import annotations

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

import json
import time
import random
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


def _save_debug_video(frames: List[np.ndarray], path: str, fps: int) -> None:
    """Save a short rollout video for debugging (no-op if frames empty)."""

    if not frames:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure HWC uint8 numpy arrays
    proc_frames: List[np.ndarray] = []
    for f in frames:
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        proc_frames.append(arr)

    imageio.mimsave(path, proc_frames, fps=max(1, int(fps)))
    print(f"[debug] saved rollout render to {path}")


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

        cpu_action = action.detach().cpu()
        nobs, rew, term, trunc, _ = envs.step(cpu_action)
        for k, v in nobs.items():
            nobs[k] = v.to(device)
        rew = rew.to(device)
        done = (term | trunc).to(device)

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


def main():
    args = tyro.cli(Args)

    run_name = args.exp_name or f"rgbppo_{args.env_name}__{args.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")

    debug_render_dir = args.debug_render_dir or os.path.join("runs", run_name, "render_debug")
    debug_render_epochs = max(0, int(args.debug_render_epochs))
    debug_render_fps = int(round(1.0 / max(args.ctrl_dt * args.action_repeat, 1e-6)))

    def _maybe_get_frame(obs: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
        if "rgb" not in obs:
            return None
        frame_t = obs["rgb"][0]
        if frame_t.is_floating_point():
            frame_t = (frame_t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        else:
            frame_t = frame_t.to(torch.uint8)
        return frame_t.detach().cpu().numpy()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device_rank}" if torch.cuda.is_available() else "cpu")

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
        gs_height=args.gs_height,
        gs_width=args.gs_width,
        gs_minibatch=int(args.gs_minibatch) if vb == "gs" else None,
        gs_disable_bg=args.gs_disable_bg,
    )

    obs, _ = envs.reset(seed=args.seed)
    for k, v in obs.items():
        obs[k] = v.to(device)

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
    done_buf = torch.zeros((num_steps, num_envs), device=device)
    val_buf = torch.zeros((num_steps, num_envs), device=device)

    next_obs = obs
    next_done = torch.zeros((num_envs,), device=device)
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    if args.eval_only and args.eval_episodes <= 0:
        raise ValueError("eval_only=True requires eval_episodes > 0")

    if not args.eval_only:
        for update in range(start_update, num_updates + 1):
            debug_frames: List[np.ndarray] = []
            record_debug = (debug_render_epochs > 0) and ((update - start_update) < debug_render_epochs)

            for step in range(num_steps):
                global_step += num_envs
                done_buf[step] = next_done

                for k in obs_buf:
                    obs_buf[k][step] = next_obs[k]

                if record_debug:
                    frame = _maybe_get_frame(next_obs)
                    if frame is not None:
                        debug_frames.append(frame)

                with torch.no_grad():
                    a, lp, ent, v = agent.get_action_and_value(next_obs)
                act_buf[step] = a
                logp_buf[step] = lp
                val_buf[step] = v

                # step env (env returns on CPU torch tensors)
                cpu_action = a.detach().cpu()
                nobs, rew, term, trunc, infos = envs.step(cpu_action)
                for k, v in nobs.items():
                    nobs[k] = v.to(device)
                rew = rew.to(device)
                done = (term | trunc).to(device).float()

                rew_buf[step] = rew
                next_obs = nobs
                next_done = done

                if record_debug:
                    frame = _maybe_get_frame(next_obs)
                    if frame is not None:
                        debug_frames.append(frame)

            # GAE
            with torch.no_grad():
                _, _, _, next_value = agent.get_action_and_value(next_obs, action=None)
                adv = torch.zeros_like(rew_buf, device=device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
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

            inds = np.arange(batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = inds[start:start + minibatch_size]
                    mb_obs = {k: v[mb] for k, v in b_obs.items()}

                    new_a, new_logp, ent, new_v = agent.get_action_and_value(mb_obs, b_act[mb])
                    ratio = (new_logp - b_logp[mb]).exp()

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
            writer.add_scalar("charts/step", global_step, global_step)

            if record_debug and debug_frames:
                vid_path = os.path.join(debug_render_dir, f"update_{update:03d}.mp4")
                _save_debug_video(debug_frames, vid_path, fps=debug_render_fps)

            if args.save_model and update % max(1, args.checkpoint_interval) == 0:
                _save_checkpoint(
                    args=args,
                    run_name=run_name,
                    agent=agent,
                    optimizer=optimizer,
                    global_step=global_step,
                    update=update,
                )

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
