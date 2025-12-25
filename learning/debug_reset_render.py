import argparse
import numpy as np
import imageio.v2 as imageio
import torch

from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_mani_wrapper import MJXManiLikeVectorEnv


def _default_airbot_body_map(reso: int):
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
        # 你已验证 GS 这边需要左右互换
        "left": (ASSETS_PATH / Reso / "right.ply").as_posix(),
        "right": (ASSETS_PATH / Reso / "left.ply").as_posix(),
        "box": (ASSETS_PATH / "green_cube.ply").as_posix(),
    }


def parse_float_list(s: str):
    # 支持 "0,0,0,..." 或 "0 0 0 ..."
    parts = [p for p in s.replace(",", " ").split(" ") if p.strip() != ""]
    return [float(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="AirbotPlayPickCube")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episode-seconds", type=float, default=3.0)
    ap.add_argument("--ctrl-dt", type=float, default=0.04)
    ap.add_argument("--action-repeat", type=int, default=1)

    ap.add_argument("--vision-backend", choices=["gs", "madrona"], default="gs")
    ap.add_argument("--include-state", action="store_true")
    ap.add_argument("--num-envs", type=int, default=1)

    # GS render params
    ap.add_argument("--gs-assets-reso", type=int, default=224)
    ap.add_argument("--gs-height", type=int, default=128)
    ap.add_argument("--gs-width", type=int, default=128)
    ap.add_argument("--gs-camera-id", type=int, default=0)

    # output
    ap.add_argument("--out", default="reset.png")
    ap.add_argument("--split-cams", action="store_true")
    ap.add_argument("--print-gripper", action="store_true")

    # step debug
    ap.add_argument("--steps", type=int, default=0, help="reset 后再执行多少步 step；0 表示只 reset 不 step")
    ap.add_argument("--action-mode", choices=["zero", "random_uniform", "constant"], default="zero")
    ap.add_argument("--constant-action", type=str, default="", help="action-mode=constant 时使用，例如 '0 0 0 0 0 0 0'")
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--save-frames", action="store_true", help="保存 reset+每步后的 rgb 帧")
    ap.add_argument("--frames-dir", type=str, default="frames")

    args = ap.parse_args()

    episode_length = int(args.episode_seconds / args.ctrl_dt)

    config_overrides = {
        "action_repeat": int(args.action_repeat),
        "ctrl_dt": float(args.ctrl_dt),
        "episode_length": int(episode_length),
    }
    raw_env = registry.load(args.env_name, config_overrides=config_overrides)

    vb = args.vision_backend.lower().strip()
    env_vision = (vb == "madrona")  # gs 时 env vision 必须关掉，由 wrapper 外部渲染
    gs_body_map = _default_airbot_body_map(args.gs_assets_reso) if vb == "gs" else None

    envs = MJXManiLikeVectorEnv(
        raw_env=raw_env,
        num_envs=args.num_envs,
        seed=args.seed,
        episode_length=episode_length,
        action_repeat=args.action_repeat,
        vision=env_vision,
        num_vision_envs=args.num_envs,
        randomization_fn=None,
        device_rank=None,
        auto_reset=False,
        include_state=bool(args.include_state),
        debug_print_obs=False,
        vision_backend=vb,
        gs_body_gaussians=gs_body_map if vb == "gs" else None,
        gs_background_ply=None,
        gs_camera_id=args.gs_camera_id,
        gs_camera_ids=None,
        gs_camera_names=None,
        gs_height=args.gs_height,
        gs_width=args.gs_width,
        gs_minibatch=16 if vb == "gs" else None,
        gs_disable_bg=False,
    )

    # ---------- reset ----------
    obs, _ = envs.reset(seed=args.seed)
    rgb = obs["rgb"][0].cpu().numpy()

    # 保存 reset 图像
    if (not args.split_cams) or (rgb.shape[-1] == 3):
        imageio.imwrite(args.out, rgb)
        print(f"[ok] wrote {args.out}, shape={rgb.shape}, dtype={rgb.dtype}")
    else:
        c = rgb.shape[-1]
        assert c % 3 == 0, f"unexpected channel count {c}"
        ncam = c // 3
        for i in range(ncam):
            out_i = args.out.replace(".png", f"_cam{i}.png")
            imageio.imwrite(out_i, rgb[:, :, 3*i:3*(i+1)])
            print(f"[ok] wrote {out_i}")

    # 打印夹爪/手指诊断
    base_env = envs._env.unwrapped
    data = envs.env_state.data

    if args.print_gripper:
        qpos = np.asarray(data.qpos)[0]
        if hasattr(base_env, "_robot_qposadr"):
            idx = np.asarray(base_env._robot_qposadr)
            finger_idx = idx[-2:]
            print("[reset qpos] finger indices:", finger_idx.tolist(), "values:", qpos[finger_idx].tolist())
        try:
            bid_left = base_env.mj_model.body("left").id
            bid_right = base_env.mj_model.body("right").id
            xpos = np.asarray(data.xpos)[0]
            print("[reset xpos] left:", xpos[bid_left].tolist())
            print("[reset xpos] right:", xpos[bid_right].tolist())
        except Exception as e:
            print("[warn] cannot query body xpos:", repr(e))

    # ---------- step loop ----------
    if args.steps <= 0:
        return

    # action dim：训练脚本里用过 envs.single_action_dim；这里优先用它
    try:
        act_dim = int(envs.single_action_dim)
    except Exception:
        # 兜底：从 state.ctrl 的维度推 nu（通常 nu==ctrl_dim）
        act_dim = int(np.asarray(data.ctrl).shape[-1])

    # constant action
    if args.action_mode == "constant":
        if not args.constant_action:
            raise ValueError("--action-mode=constant 需要提供 --constant-action")
        a = np.asarray(parse_float_list(args.constant_action), dtype=np.float32)
        if a.size != act_dim:
            raise ValueError(f"constant-action dim mismatch: got {a.size}, expected {act_dim}")
        const_action = torch.from_numpy(a).view(1, -1)
    else:
        const_action = None

    # 取 finger qpos indices（用于打印）
    finger_idx = None
    if hasattr(base_env, "_robot_qposadr"):
        idx = np.asarray(base_env._robot_qposadr)
        finger_idx = idx[-2:]

    # 可选保存每步帧
    if args.save_frames:
        import os
        os.makedirs(args.frames_dir, exist_ok=True)
        # 保存 reset 帧
        imageio.imwrite(f"{args.frames_dir}/frame_0000.png", rgb[:, :, :3] if rgb.shape[-1] >= 3 else rgb)

    for t in range(1, args.steps + 1):
        # 生成 action_raw（你送进 env.step 的）
        if args.action_mode == "zero":
            action_raw = torch.zeros((envs.num_envs, act_dim), dtype=torch.float32)
        elif args.action_mode == "random_uniform":
            action_raw = (torch.rand((envs.num_envs, act_dim), dtype=torch.float32) * 2.0 - 1.0)
        elif args.action_mode == "constant":
            action_raw = const_action.repeat(envs.num_envs, 1).clone()
        else:
            raise RuntimeError("unknown action mode")

        # 记录 step 前 ctrl/qpos
        ctrl0 = np.asarray(envs.env_state.data.ctrl)[0].copy()
        qpos0 = np.asarray(envs.env_state.data.qpos)[0].copy()

        # 环境等效执行的 action（你的 wrapper 内部会 clamp；这里显式算出来便于对照）
        action_applied = torch.clamp(action_raw, -1.0, 1.0)

        # step
        nobs, rew, term, trunc, info = envs.step(action_raw)

        ctrl1 = np.asarray(envs.env_state.data.ctrl)[0].copy()
        qpos1 = np.asarray(envs.env_state.data.qpos)[0].copy()

        dctrl = ctrl1 - ctrl0
        dqpos = qpos1 - qpos0

        if (t % args.print_every) == 0:
            # 只打印 env0，避免刷屏
            ar0 = action_raw[0].cpu().numpy()
            aa0 = action_applied[0].cpu().numpy()

            msg = [
                f"[t={t:04d}]",
                f"action_raw[0]={np.array2string(ar0, precision=3, suppress_small=True)}",
                f"action_applied[0]={np.array2string(aa0, precision=3, suppress_small=True)}",
                f"|dctrl|={float(np.linalg.norm(dctrl)):.6f}",
                f"|dqpos|={float(np.linalg.norm(dqpos)):.6f}",
            ]
            if finger_idx is not None:
                msg.append(f"finger_qpos0={qpos0[finger_idx].tolist()} -> finger_qpos1={qpos1[finger_idx].tolist()}")
                msg.append(f"dfinger={(qpos1[finger_idx]-qpos0[finger_idx]).tolist()}")
            print("  ".join(msg))

        # 可选保存每步帧（用 nobs 里的 rgb）
        if args.save_frames:
            rgb_t = nobs["rgb"][0].cpu().numpy()
            imageio.imwrite(f"{args.frames_dir}/frame_{t:04d}.png", rgb_t[:, :, :3] if rgb_t.shape[-1] >= 3 else rgb_t)

    print("[ok] step debug finished.")


if __name__ == "__main__":
    main()
