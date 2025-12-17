from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Mapping

import numpy as np
import jax
import jax.numpy as jnp

import torch
import torch.utils.dlpack as tpack

from mujoco_playground._src import wrapper as mjx_wrapper
from mujoco_playground._src import mjx_env


# ---------- dlpack helpers ----------
def _jax_to_torch(x: jax.Array) -> torch.Tensor:
    return tpack.from_dlpack(x)


def _torch_to_jax(x: torch.Tensor) -> jax.Array:
    from jax.dlpack import from_dlpack
    return from_dlpack(x)


# ---------- optional DISCOVERSE GS ----------
_HAS_GS = True
try:
    from discoverse.gaussian_renderer.batch_splat import BatchSplatConfig, BatchSplatRenderer
except Exception:
    _HAS_GS = False
    BatchSplatConfig = None  # type: ignore
    BatchSplatRenderer = None  # type: ignore


@dataclass
class MJXVisionSpec:
    rgb_shape: Tuple[int, int, int]  # (H, W, C)


class MJXManiLikeVectorEnv:
    """
    A ManiSkillVectorEnv-like wrapper with MJX env backend.

    Outputs obs dict:
      - obs["rgb"]: uint8 [B,H,W,3]
      - obs["state"]: float32 [B,D] (optional)

    vision_backend:
      - "madrona": expect env_state.obs already includes pixels (needs MadronaWrapper)
      - "gs":      render pixels externally via BatchSplatRenderer (torch)
    """

    def __init__(
        self,
        raw_env,
        num_envs: int,
        seed: int,
        episode_length: int,
        action_repeat: int = 1,
        vision: bool = True,
        num_vision_envs: Optional[int] = None,
        randomization_fn=None,
        device_rank: Optional[int] = None,
        auto_reset: bool = True,
        include_state: bool = True,
        debug_print_obs: bool = False,

        # NEW
        vision_backend: str = "madrona",  # "madrona" | "gs"
        gs_body_gaussians: Optional[Mapping[str, str]] = None,
        gs_background_ply: Optional[str] = None,
        gs_camera_id: int = 0,
        gs_height: int = 60,
        gs_width: int = 80,
        gs_minibatch: Optional[int] = None,
        gs_disable_bg: bool = False,
    ):
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.auto_reset = bool(auto_reset)
        self.include_state = bool(include_state)
        self.debug_print_obs = bool(debug_print_obs)

        self.vision_backend = (vision_backend or "madrona").lower().strip()

        # Torch device: align with JAX GPU if device_rank provided
        if torch.cuda.is_available() and device_rank is not None:
            self.torch_device = torch.device(f"cuda:{int(device_rank)}")
        else:
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Under GS backend: DO NOT enable env vision (avoid MadronaWrapper)
        if self.vision_backend == "gs":
            vision = False

        # Wrap MJX env
        self._env = mjx_wrapper.wrap_for_brax_training(
            raw_env,
            vision=bool(vision),
            num_vision_envs=(int(num_vision_envs) if num_vision_envs is not None else self.num_envs),
            episode_length=int(episode_length),
            action_repeat=int(action_repeat),
            randomization_fn=randomization_fn,
            # if caller accidentally passed vision=True with gs, we still avoid Madrona vision here:
            vision_backend=("external" if self.vision_backend == "gs" else "madrona"),
        )

        self._reset_jit = jax.jit(self._env.reset)
        self._step_jit = jax.jit(self._env.step)

        # RNG
        self.key = jax.random.PRNGKey(self.seed)
        if device_rank is not None:
            dev = jax.devices("gpu")[int(device_rank)]
            self.key = jax.device_put(self.key, dev)

        self.env_state = None

        # MJ model access (needed for fovy + body name mapping in GS renderer init)
        self.mj_model = getattr(raw_env, "mj_model", None)
        if self.mj_model is None and hasattr(raw_env, "unwrapped"):
            self.mj_model = getattr(raw_env.unwrapped, "mj_model", None)

        # GS settings
        self._gs_body_gaussians = dict(gs_body_gaussians) if gs_body_gaussians is not None else {}
        self._gs_background_ply = gs_background_ply
        self._gs_camera_id = int(gs_camera_id)
        self._gs_height = int(gs_height)
        self._gs_width = int(gs_width)
        self._gs_minibatch = gs_minibatch
        self._gs_disable_bg = bool(gs_disable_bg)

        self._gs_renderer: Optional["BatchSplatRenderer"] = None
        self._bg_img_cache: Optional[torch.Tensor] = None

        # vision spec
        self._vision_spec = MJXVisionSpec(rgb_shape=(self._gs_height, self._gs_width, 3))

        # Do a warm reset to infer obs structure
        obs0, _ = self.reset(seed=self.seed)
        self.single_action_dim = int(self._env.env.unwrapped.action_size)

        # Keep a sample state dim if included
        self._state_dim = int(obs0["state"].shape[-1]) if ("state" in obs0) else None

    # ---------- GS renderer ----------
    def _ensure_gs_renderer(self):
        if self._gs_renderer is not None:
            return
        if not _HAS_GS:
            raise ImportError(
                "discoverse.gaussian_renderer.batch_splat import failed. "
                "Please ensure DISCOVERSE + gsplat dependencies are installed in this env."
            )
        if self.mj_model is None:
            raise RuntimeError("GS backend needs env.mj_model to resolve body/camera metadata.")
        if (not self._gs_body_gaussians) and (self._gs_background_ply is None):
            raise ValueError("GS backend requires gs_body_gaussians or gs_background_ply.")

        cfg = BatchSplatConfig(
            body_gaussians=self._gs_body_gaussians,
            background_ply=self._gs_background_ply,
            minibatch=self._gs_minibatch,
        )
        self._gs_renderer = BatchSplatRenderer(cfg, mj_model=self.mj_model)

    def _get_bg_img(self, B: int, Ncam: int, H: int, W: int) -> Optional[torch.Tensor]:
        if self._gs_disable_bg:
            return None
        # cache shape-specific bg tensor
        if self._bg_img_cache is None or self._bg_img_cache.shape != (B, Ncam, H, W, 3):
            self._bg_img_cache = torch.ones((B, Ncam, H, W, 3), dtype=torch.float32, device=self.torch_device)
        return self._bg_img_cache

    @torch.no_grad()
    def _render_gs_rgb(self, state) -> torch.Tensor:
        self._ensure_gs_renderer()
        assert self._gs_renderer is not None

        data = state.data
        body_pos = _jax_to_torch(data.xpos).to(self.torch_device)     # (B,Nbody,3)
        body_quat = _jax_to_torch(data.xquat).to(self.torch_device)   # (B,Nbody,4) wxyz
        gsb = self._gs_renderer.batch_update_gaussians(body_pos, body_quat)

        cam_pos = _jax_to_torch(data.cam_xpos).to(self.torch_device)   # (B,Ncam,3)
        cam_xmat = _jax_to_torch(data.cam_xmat).to(self.torch_device)  # (B,Ncam,3,3) or (B,Ncam,9)

        if cam_xmat.ndim == 3:  # (B,Ncam,9)
            pass
        elif cam_xmat.ndim == 4 and cam_xmat.shape[-2:] == (3, 3):
            # renderer accepts (B,Ncam,3,3) in your notebook; keep as-is
            pass
        else:
            raise RuntimeError(f"Unexpected cam_xmat shape: {tuple(cam_xmat.shape)}")

        # fovy broadcast to (B,Ncam)
        fovy = np.asarray(self.mj_model.cam_fovy, dtype=np.float32)[None, :]  # (1,Ncam)
        B = int(cam_pos.shape[0])
        fovy = np.broadcast_to(fovy, (B, fovy.shape[1]))

        bg = self._get_bg_img(B, int(fovy.shape[1]), self._gs_height, self._gs_width)

        rgb, _depth = self._gs_renderer.batch_env_render(
            gsb,
            cam_pos,
            cam_xmat,
            self._gs_height,
            self._gs_width,
            fovy,
            bg,
        )  # rgb: (B,Ncam,H,W,3) float32

        # choose one camera for PPO
        cam_id = self._gs_camera_id
        rgb = rgb[:, cam_id, ...]  # (B,H,W,3)

        # float->uint8
        rgb_u8 = (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return rgb_u8

    # ---------- obs extraction ----------
    def _extract_obs(self, state) -> Dict[str, torch.Tensor]:
        obs: Dict[str, torch.Tensor] = {}

        if self.vision_backend == "gs":
            obs["rgb"] = self._render_gs_rgb(state)
        else:
            # Madrona path: expect dict obs with a key 'rgb' (or adjust if your env uses another key)
            if not isinstance(state.obs, dict) or ("rgb" not in state.obs):
                raise RuntimeError(
                    "vision_backend='madrona' requires env_state.obs to be dict containing key 'rgb'. "
                    "If you want GS rendering, use --vision-backend gs."
                )
            rgb = _jax_to_torch(state.obs["rgb"]).to(self.torch_device)
            if rgb.dtype != torch.uint8:
                # tolerate float [0,1]
                if rgb.is_floating_point():
                    rgb = (rgb.clamp(0, 1) * 255.0).to(torch.uint8)
                else:
                    rgb = rgb.to(torch.uint8)
            obs["rgb"] = rgb

        if self.include_state:
            if isinstance(state.obs, dict) and ("state" in state.obs):
                obs["state"] = _jax_to_torch(state.obs["state"]).to(self.torch_device).to(torch.float32)
            elif not isinstance(state.obs, dict):
                obs["state"] = _jax_to_torch(state.obs).to(self.torch_device).to(torch.float32)
            # else: dict but no "state" => omit

        return obs

    # ---------- API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.key = jax.random.PRNGKey(int(seed))
        self.key, sub = jax.random.split(self.key)
        keys = jax.random.split(sub, self.num_envs)
        self.env_state = self._reset_jit(keys)

        if self.debug_print_obs:
            print("[DEBUG] env_state.obs type:", type(self.env_state.obs))
            if isinstance(self.env_state.obs, dict):
                print("[DEBUG] env_state.obs keys:", list(self.env_state.obs.keys()))
            else:
                a = np.asarray(self.env_state.obs)
                print("[DEBUG] env_state.obs shape:", a.shape, a.dtype)

        obs = self._extract_obs(self.env_state)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.0, 1.0)
        act_jax = _torch_to_jax(actions)

        next_state = self._step_jit(self.env_state, act_jax)
        self.env_state = next_state

        obs = self._extract_obs(self.env_state)
        rew = _jax_to_torch(next_state.reward).to(self.torch_device).to(torch.float32)
        done = _jax_to_torch(next_state.done).to(self.torch_device).to(torch.bool)

        terminations = done
        truncations = torch.zeros_like(done, dtype=torch.bool, device=self.torch_device)
        infos: Dict[str, Any] = {}
        return obs, rew, terminations, truncations, infos
