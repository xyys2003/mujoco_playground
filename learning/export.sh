#!/usr/bin/env bash
# 导出 Frank (PandaPickCubeOrientation) 和 AirbotPlayPickCube 的 PPO 模型为 ONNX

set -e  # 任何一步出错就中止脚本


echo "===== Start exporting PPO checkpoints to ONNX ====="

# ===== PandaPickCubeOrientation / Frank PPO → ONNX =====
echo "[1/6] Export Frank 1765452326 ..."
python export_frank_ppo_onnx.py \
  --checkpoint /root/data/code/junzhe/learning/notebooks/runs/PandaPickCubeOrientation__ppo_torch__1__1765452326/final_ckpt.pt \
  --output /root/data/code/junzhe/mj2/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/panda_orient_state_ppo701.onnx

echo "[2/6] Export Frank 1765452188 ..."
python export_frank_ppo_onnx.py \
  --checkpoint /root/data/code/junzhe/learning/notebooks/runs/PandaPickCubeOrientation__ppo_torch__1__1765452188/final_ckpt.pt \
  --output /root/data/code/junzhe/mj2/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/panda_orient_state_ppo702.onnx

echo "[3/6] Export Frank 1765452432 ..."
python export_frank_ppo_onnx.py \
  --checkpoint /root/data/code/junzhe/learning/notebooks/runs/PandaPickCubeOrientation__ppo_torch__1__1765452432/final_ckpt.pt \
  --output /root/data/code/junzhe/mj2/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/panda_orient_state_ppo703.onnx

# ===== AirbotPlayPickCube PPO → ONNX =====
echo "[4/6] Export Airbot 1765452002 ..."
python export_airbot_ppo_onnx.py \
  --checkpoint /root/data/code/junzhe/learning/notebooks/runs/AirbotPlayPickCube__ppo_torch__1__1765452002/final_ckpt.pt \
  --output /root/data/code/junzhe/mj2/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/airbot_state_ppo602.onnx

echo "[5/6] Export Airbot 1765452077 ..."
python export_airbot_ppo_onnx.py \
  --checkpoint /root/data/code/junzhe/learning/notebooks/runs/AirbotPlayPickCube__ppo_torch__1__1765452077/final_ckpt.pt \
  --output /root/data/code/junzhe/mj2/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/airbot_state_ppo603.onnx

echo "[6/6] Export Airbot 1765452289 ..."
python export_airbot_ppo_onnx.py \
  --checkpoint /root/data/code/junzhe/learning/notebooks/runs/AirbotPlayPickCube__ppo_torch__1__1765452289/final_ckpt.pt \
  --output /root/data/code/junzhe/mj2/mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/airbot_state_ppo604.onnx

echo "===== All exports finished successfully. ====="
