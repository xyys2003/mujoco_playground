#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script converted from manipulation.ipynb.
Contains all code cells that appear before the "Dexterous Manipulation" section.
Original notebook: MuJoCo Playground manipulation example.
"""


# ===== Cell 4 =====
#@title Install pre-requisites
# !pip install mujoco  # (original notebook shell command)
# !pip install mujoco_mjx  # (original notebook shell command)
# !pip install brax  # (original notebook shell command)


# ===== Cell 5 =====
# @title Check if MuJoCo installation was successful

import distutils.util
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_EXTERNAL_PLATFORM_CONFIG_DIR",
                      "/data/yufei/egl_external_platform.d")
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
EGL_EXTERNAL_PLATFORM_CONFIG = r"""{
  "file_format_version" : "1.0.0",
  "ICD" : {
    "library_path" : "libEGL_nvidia.so.0"
  }
}"""

# Use a writable directory instead of /usr/lib...
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


# Make sure MuJoCo can create a GPU context. If this fails, the most common
# cause is that your LD_LIBRARY_PATH needs to be configured so that the Nvidia
# EGL library can be resolved. On Colab you can run the following command to
# locate the library:
#
#   find /usr -name 'libEGL_nvidia.so.0'
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

print('Installation successful.')

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


# ===== Cell 6 =====
# @title Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np


import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


# ===== Cell 7 =====
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp


# ===== Cell 8 =====
#@title Install MuJoCo Playground
# !pip install playground  # (original notebook shell command)


# ===== Cell 9 =====
#@title Import The Playground

from mujoco_playground import wrapper
from mujoco_playground import registry


# ===== Cell 11 =====
registry.manipulation.ALL_ENVS


# ===== Cell 13 =====
import jax
print(jax.devices())

env_name = 'PandaPickCubeOrientation'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)


# ===== Cell 14 =====
env_cfg


# ===== Cell 16 =====
from mujoco_playground.config import manipulation_params
ppo_params = manipulation_params.brax_ppo_config(env_name)
ppo_params


# ===== Cell 18 =====
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")

  plt.errorbar(
      x_data,
      y_data,
      yerr=y_dataerr,
      ecolor="black",
      capsize=5,
  )
  plt.scatter(x_data, y_data)
  plt.show()


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    seed=1
)


# ===== Cell 19 =====
make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")


# ===== Cell 21 =====
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))


# ===== Cell 22 =====
rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
  state = jit_reset(rng)
  rollout.append(state)
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)

render_every = 1
rewards = [s.reward for s in rollout]

video_path = os.path.join(os.path.dirname(__file__),
                          "panda_pick_cube_orientation.mp4")

try:
    frames = env.render(rollout[::render_every])
    media.write_video(
        video_path,
        frames,
        fps=1.0 / env.dt / render_every,
    )
    print(f"Video saved to: {video_path}")
except Exception as e:
    print("[WARN] Rendering failed:", repr(e))
    print("Training and rollout are done, but video could not be rendered.")



if __name__ == "__main__":
    # The notebook code runs at import time;
    # importing this script will execute everything above.
    # You can put extra custom logic here if desired.
    pass
