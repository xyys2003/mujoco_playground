from mujoco_playground import registry
import os 
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_EXTERNAL_PLATFORM_CONFIG_DIR",
                      "/data/yufei/egl_external_platform.d")
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/data/yufei/jax_cache"
# Reuse all PPO plumbing from the state-based script.
import state_ppo as _ppo

_AIRBOT_ENV = "AirbotPlayPickCube"

if _AIRBOT_ENV not in registry.ALL_ENVS:
    available = ", ".join(registry.ALL_ENVS)
    raise RuntimeError(
        f"Requested env '{_AIRBOT_ENV}' is not registered. "
        "Please update mujoco_playground or choose one of: "
        f"{available}"
    )

# Force the environment globals to point to Airbot Play pick.

_ppo.env_name = _AIRBOT_ENV
_ppo.env = registry.load(_ppo.env_name)
_ppo.env_cfg = registry.get_default_config(_ppo.env_name)

# Keep the CLI default aligned with the Airbot Play entrypoint.
_ppo.Args.env_id = _AIRBOT_ENV


if __name__ == "__main__":
    _ppo.main()