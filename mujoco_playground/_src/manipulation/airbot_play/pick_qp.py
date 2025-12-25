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
      # Debug toggles (default: fixed positions for staged learning)
      debug_fixed_box=True,        # keep box at home keyframe position
      debug_fixed_target=True,     # keep target at pickup2 keyframe box position
      debug_init_keyframe="home",  # initial robot pose/ctrl: "home" (default) or "pick1"
      reached_box_thresh=0.012,          # reached_box threshold on gripper-box distance (meters)
      reward_config=config_dict.create(
    scales=config_dict.create(
        # --- Stage-0: go from home -> pick1 (joint-space target) ---
        gripper_pick1=4.0,

        # --- Stage-1: after reached_box, go pick1 -> pickup2 ---
        box_target=30.0,
        arm_pickup2=2.0,

        # Safety
        no_floor_collision=0.25,

        # --- Logging-only metrics (zero scale, do not affect reward) ---
        gripper_box=0.0,       # keep old metric for TB / stage switch
        gripper_dist=0.0,
        reached_box=0.0,
        stage=0.0,
        gripper_pick1_dist=0.0,
        arm_pickup2_err=0.0,
        pos_err=0.0,
        rot_err=0.0,

        # Keep for compatibility (disabled)
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

    # --- Keyframe targets for staged debugging ---
    # pick1: pre-grasp pose; pickup2: post-grasp/lift pose.
    try:
      _kf_pick1 = jp.array(self._mj_model.keyframe("pick1").qpos)
      _kf_pickup2 = jp.array(self._mj_model.keyframe("pickup2").qpos)
    except Exception as exc:
      raise ValueError(
          "Keyframes 'pick1' and/or 'pickup2' not found in XML. "
          "Please ensure they exist in <keyframe>."
      ) from exc

    self._pick1_arm_qpos = _kf_pick1[self._robot_arm_qposadr]

    # Precompute gripper site position at pick1 (for stage-0 position target reward)
    _data_pick1 = mjx_env.make_data(
        self._mj_model,
        qpos=_kf_pick1,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._mj_model.keyframe("pick1").ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    _data_pick1 = mjx.forward(self._mjx_model, _data_pick1)
    self._pick1_gripper_pos = _data_pick1.site_xpos[self._gripper_site]
    self._pickup2_arm_qpos = _kf_pickup2[self._robot_arm_qposadr]

    # Fixed target position from pickup2 keyframe (box freejoint translation)
    self._pickup2_box_pos = jp.array(
        _kf_pickup2[self._obj_qposadr : self._obj_qposadr + 3],
        dtype=jp.float32,
    )
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_box"]
    ]

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
    if bool(self._config.get("debug_fixed_box", False)):
      box_pos = jp.array(self._init_obj_pos, dtype=jp.float32)
    else:
      box_pos = (
          jax.random.uniform(
              rng_box,
              (3,),
              minval=jp.array([-0.1, -0.1, 0.0]),
              maxval=jp.array([0.1, 0.1, 0.0]),
          )
          + self._init_obj_pos
      )

    # initialize target position
    if bool(self._config.get("debug_fixed_target", False)):
      target_pos = jp.array(self._pickup2_box_pos, dtype=jp.float32)
    else:
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
    # Optionally start robot from a different keyframe (default: "home").
    init_kf = str(self._config.get("debug_init_keyframe", "home"))
    try:
      init_q_base = jp.array(self._mj_model.keyframe(init_kf).qpos)
      init_ctrl_base = self._mj_model.keyframe(init_kf).ctrl
    except Exception:
      init_q_base = jp.array(self._init_q)
      init_ctrl_base = self._init_ctrl

    # Always overwrite the box freejoint translation with our box_pos.
    init_q = (
        init_q_base
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=init_ctrl_base,
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
    metrics = {
    "out_of_bounds": jp.array(0.0, dtype=jp.float32),
    "nan_qpos": jp.array(0.0, dtype=jp.float32),
    "nan_qvel": jp.array(0.0, dtype=jp.float32),
    "nan_raw": jp.array(0.0, dtype=jp.float32),
    "done_fail": jp.array(0.0, dtype=jp.float32),
    **{k: jp.array(0.0, dtype=jp.float32) for k in self._config.reward_config.scales.keys()},
}
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
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
    # ---------- common geometry ----------
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]

    # distance-based "old" metric (kept for TB and stage switch)
    gripper_dist = jp.linalg.norm(box_pos - gripper_pos)
    gripper_box = 1 - jp.tanh(5 * gripper_dist)

    # Stage switch: reached_box becomes sticky once gripper is close enough.
    reach_thresh = float(self._config.get("reached_box_thresh", 0.012))
    reached_now = (gripper_dist < reach_thresh).astype(float)
    info["reached_box"] = 1.0 * jp.maximum(info["reached_box"], reached_now)

    stage = info["reached_box"]          # 0.0 -> stage0, 1.0 -> stage1
    stage0 = 1.0 - stage
    stage1 = stage

    # Arm joint positions (used for stage-1 arm shaping / logging)
    arm_qpos = data.qpos[self._robot_arm_qposadr]

    # ---------- Stage 0: home -> pick1 (position-space target) ----------
    gripper_pick1_dist = jp.linalg.norm(gripper_pos - self._pick1_gripper_pos)
    gripper_pick1 = 1.0 - jp.tanh(5.0 * gripper_pick1_dist)

    # ---------- Stage 1: pick1 -> pickup2 ----------
    # Box to target mocap (target_pos is pickup2 box pos when debug_fixed_target=True)
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
    box_target = 1.0 - jp.tanh(5.0 * (0.9 * pos_err + 0.1 * rot_err))

    # Optional: keep arm near pickup2 pose after stage switch
    arm_pickup2_err = jp.linalg.norm(arm_qpos - self._pickup2_arm_qpos)
    arm_pickup2 = 1.0 - jp.tanh(arm_pickup2_err)

    # ---------- Safety: floor collision ----------
    hand_floor_collision = [
    data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
    for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    rewards = {
    # Reward terms
    "gripper_pick1": gripper_pick1 * stage0,
    "box_target": box_target * stage1,
    "arm_pickup2": arm_pickup2 * stage1,
    "no_floor_collision": no_floor_collision,

    # Logging-only (scales are 0.0)
    "gripper_box": gripper_box,
    "gripper_dist": gripper_dist,
    "reached_box": info["reached_box"],
    "stage": stage,
    "gripper_pick1_dist": gripper_pick1_dist,
    "arm_pickup2_err": arm_pickup2_err,
    "pos_err": pos_err,
    "rot_err": rot_err,

    # Keep disabled legacy term for compatibility
    "robot_target_qpos": jp.array(0.0, dtype=jp.float32),
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