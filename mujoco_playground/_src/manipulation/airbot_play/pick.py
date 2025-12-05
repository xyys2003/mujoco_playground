from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.airbot_play import airbot_play
from mujoco_playground._src.mjx_env import State
from ml_collections import config_dict

def default_config() -> config_dict.ConfigDict:
    """Returns the default config for bring_to_target tasks."""
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=150,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Gripper goes to the box.
                gripper_box=4.0,
                # Box goes to the target mocap.
                box_target=8.0,
                # Do not collide the gripper with the floor.
                no_floor_collision=0.25,
                # Arm stays close to target pose.
                robot_target_qpos=0.3,
            )
        ),
        impl='jax',
        nconmax=24 * 2048,
        njmax=128,
    )
    return config

class AirbotPlayPick(airbot_play.AirbotPlayBase):
    def __init__(
        self,
        xml_path: Optional[epath.Path] = None,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        xml_path = xml_path or (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "airbot_play"
            / "xmls"
            / "airbot_play.xml"
        )

        xml_path = epath.Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(
                "Airbot Play model file not found. "
                "Place the Airbot Play MJX XML at "
                f"'{xml_path}' or pass an explicit xml_path to AirbotPlayPick."
            )

        super().__init__(xml_path, config, config_overrides)
        self._post_init(keyframe="home")

    def _post_init(self, keyframe: str):
        self._gripper_site = self._mj_model.site("endpoint").id

        self._init_q = self._mj_model.keyframe(keyframe).qpos
        self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

    def reset(self, rng: jax.Array) -> State:
        # rng, rng_box, rng_target = jax.random.split(rng, 3)

        # initialize data
        init_q = jp.array(self._init_q)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        # initialize env state and info
        metrics = {}
        info = {"rng": rng}
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2, dtype=jp.float32)
        state = State(data, obs, reward, done, metrics, info)
        return state

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_mat = data.site_xmat[self._gripper_site].ravel()
        obs = jp.concatenate([
            data.qpos,
            data.qvel,
            gripper_pos,
            gripper_mat[3:],
        ])

        return obs

    def step(self, state: State, action: jax.Array) -> State:
        delta = action * self._action_scale
        ctrl = state.data.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        reward = jp.zeros(1, dtype=jp.float32)
        done = jp.zeros(1, dtype=jp.float32)

        obs = self._get_obs(data, state.info)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state
