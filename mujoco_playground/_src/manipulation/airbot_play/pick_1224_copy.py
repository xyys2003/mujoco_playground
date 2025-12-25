"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.airbot_play import airbot_play
from mujoco_playground._src.mjx_env import State
import numpy as np


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
            gripper_box=8.0,
            box_target=40.0,
            no_floor_collision=0.0,
            robot_target_qpos=0.0,
        )
    ),

      impl='jax',
      nconmax=24 * 2048,
      njmax=128,
  )
  return config


class AirbotPlayPickCube(airbot_play.AirbotPlayBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "airbot_play"
        / "xmls"
        / "mjx_single_cube.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="home")
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_box"]
    ]
    self._lf_box_sensor = self._mj_model.sensor("left_finger_pad_box_found").id
    self._rf_box_sensor = self._mj_model.sensor("right_finger_pad_box_found").id
    self._lf_box_adr = self._mj_model.sensor_adr[self._lf_box_sensor]
    self._rf_box_adr = self._mj_model.sensor_adr[self._rf_box_sensor]
    self._gripper_act = self._mj_model.actuator("gripper").id


  def _post_init(self, obj_name: str, keyframe: str):
    all_joints = airbot_play._ARM_JOINTS + airbot_play._FINGER_JOINTS
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in airbot_play._ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("endpoint").id
    self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
    self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
    self._hand_geom = self._mj_model.geom("hand_box").id
    self._obj_body = self._mj_model.body(obj_name).id
    self._obj_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body(obj_name).jntadr[0]
    ]
    self._mocap_target = self._mj_model.body("mocap_target").mocapid
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_obj_pos = jp.array(
        self._init_q[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=jp.float32,
    )
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # intialize box position
    # box_pos = (
    #     jax.random.uniform(
    #         rng_box,
    #         (3,),
    #         minval=jp.array([-0.1, -0.1, 0.0]),
    #         maxval=jp.array([0.1, 0.1, 0.0]),
    #     )
    #     + self._init_obj_pos
    # )
    box_pos = self._init_obj_pos

    # initialize target position
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.1, -0.1, 0.1]),
            maxval=jp.array([0.1, 0.1, 0.2]),
        )
        + self._init_obj_pos
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if self._sample_orientation:
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
      target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

    # initialize data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # set target mocap position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    data = mjx.forward(self._mjx_model, data)

    # initialize env state and info
    # metrics = {
    #     "out_of_bounds": jp.array(0.0, dtype=float),
    #     **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    # }
    metrics = {
    "out_of_bounds": jp.array(0.0, dtype=jp.float32),
    "nan_qpos": jp.array(0.0, dtype=jp.float32),
    "nan_qvel": jp.array(0.0, dtype=jp.float32),
    "nan_raw": jp.array(0.0, dtype=jp.float32),
    "done_fail": jp.array(0.0, dtype=jp.float32),
    **{k: jp.array(0.0, dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
}

    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0,    "grasped": jp.array(0.0, dtype=jp.float32)}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

#   def step(self, state: State, action: jax.Array) -> State:
#     delta = action * self._action_scale
#     ctrl = state.data.ctrl + delta
#     ctrl = jp.clip(ctrl, self._lowers, self._uppers)

#     data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
#     data = mjx.forward(self._mjx_model, data)

#     raw_rewards = self._get_reward(data, state.info)
#     rewards = {
#         k: v * self._config.reward_config.scales[k]
#         for k, v in raw_rewards.items()
#     }
#     # reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
#     # box_pos = data.xpos[self._obj_body]
#     # out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
#     # out_of_bounds |= box_pos[2] < 0.0
#     # done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | jp.isnan(reward)
#     # done = done.astype(float)

#     # reward = jp.where(jp.isnan(reward), -1e4, reward)
#     raw_sum = sum(rewards.values())                 # 未 clip
#     reward = jp.clip(raw_sum, -1e-4, 1e-4)

#     box_pos = data.xpos[self._obj_body]
#     out_of_bounds = jp.any(jp.abs(box_pos) > 1.0) | (box_pos[2] < 0.0)

#     nan_qpos = jp.isnan(data.qpos).any()
#     nan_qvel = jp.isnan(data.qvel).any()
#     nan_raw  = jp.isnan(raw_sum)                    # 检测更有意义：clip 前
#     # nan_reward = jp.isnan(reward)                 # 这行可留可删；理论上与 nan_raw 等价

#     bad_state = nan_qpos | nan_qvel | nan_raw
#     done_fail = out_of_bounds | bad_state
#     done = done_fail.astype(float)

#     reward = jp.where(done_fail, 0.0, reward)

#     state.metrics.update(
#         **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
#     )

#     obs = self._get_obs(data, state.info)
#     state = State(data, obs, reward, done, state.metrics, state.info)

#     return state

  def step(self, state: State, action: jax.Array) -> State:
    if "steps" in state.info:
        first = (state.info["steps"] == 0)  # shape: () or (B,)
        info0 = dict(state.info)
        info0["reached_box"] = jp.where(first, 0.0, info0["reached_box"])
        info0["grasped"] = jp.where(first, 0.0, info0["grasped"])
        state = state.replace(info=info0)
    # 1) 控制：delta -> ctrl，并做 ctrlrange clip（这部分保持你的原逻辑即可）
    delta = action * self._action_scale
    ctrl = jp.clip(state.data.ctrl + delta, self._lowers, self._uppers)

    # 2) 仿真推进
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    data = mjx.forward(self._mjx_model, data)

    # 3) reward（建议先算 raw_sum，再 clip；nan 检测放在 clip 前）
    raw_rewards = self._get_reward(data, state.info)
    rewards = {k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()}

    raw_sum = sum(rewards.values())                         # 可能是标量或 (B,)
    reward = jp.clip(raw_sum, -1e4, 1e4)                    # 这里一定要是 1e4 级别，不要写成 1e-4

    # 4) 失败终止判定（按“每个 env”计算，axis=-1 保证 batched 正确）
    box_pos = data.xpos[self._obj_body]                     # (3,) 或 (B,3)
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0, axis=-1) | (box_pos[..., 2] < 0.0)

    nan_qpos = jp.any(jp.isnan(data.qpos), axis=-1)         # ( ) 或 (B,)
    nan_qvel = jp.any(jp.isnan(data.qvel), axis=-1)
    nan_raw  = jp.isnan(raw_sum)                            # raw_sum 为 NaN 才是最关键的“源头信号”

    bad_state = nan_qpos | nan_qvel | nan_raw
    done_fail = out_of_bounds | bad_state

    # 5) 如果你未来要加成功终止（现在没有就保持 False）
    # done_success = (pos_err < thresh) & (rot_err < thresh2)  # shape 与 done_fail 一致
    done_success = jp.zeros_like(done_fail, dtype=jp.bool_)

    done = (done_fail | done_success).astype(jp.float32)

    # 6) 对失败终止的 reward 做“去污染”：置 0（或一个小惩罚）
    reward = jp.where(done_fail, 0.0, reward)

    # 7) metrics：把“比例类指标”写进去，后续 vector wrapper 会做 mean -> TensorBoard
    state.metrics.update(
        **raw_rewards,
        out_of_bounds=out_of_bounds.astype(jp.float32),
        nan_qpos=nan_qpos.astype(jp.float32),
        nan_qvel=nan_qvel.astype(jp.float32),
        nan_raw=nan_raw.astype(jp.float32),
        done_fail=done_fail.astype(jp.float32),
    )

    obs = self._get_obs(data, state.info)
    return State(data, obs, reward, done, state.metrics, state.info)


  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos[self._robot_qposadr],
        data.qvel[self._robot_qposadr],
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs