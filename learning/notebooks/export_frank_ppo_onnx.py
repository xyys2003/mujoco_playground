#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export AirbotPlayPickCube PPO (PyTorch checkpoint) to ONNX.

从 notebooks/state_ppo.py 里导入 Agent 和 make_playground_vec_env，
加载 ckpt_XXXX.pt，然后导出一个只做 obs->action 的 ONNX 模型。
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn

# 确保 notebooks 目录是 package: 有 __init__.py
# 你之前已经有 /learning/notebooks/state_ppo.py 了
from notebooks.state_ppo_frank import Agent, make_playground_vec_env


class PolicyWrapper(nn.Module):
    """把你的 Agent 包一层，提供 forward(obs)->action 接口."""

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs 期望是 [B, obs_dim]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        # 使用确定性策略（等价于 rollout 里 deterministic=True）
        actions = self.agent.get_action(obs, deterministic=True)
        return actions


def build_policy_from_checkpoint(checkpoint_path: str,
                                 device: torch.device) -> tuple[nn.Module, int, int]:
    """构建 env + Agent + PolicyWrapper，并加载权重."""
    # 1) 用 Playground 的 vec env 拿到观察维度和动作维度（保证与训练一致）
    envs = make_playground_vec_env(num_envs=1, seed=0)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    print(f"[INFO] obs_dim = {obs_dim}, act_dim = {act_dim}")

    # 2) 构建 Agent 并加载 checkpoint
    agent = Agent(envs).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    # 3) 包成 PolicyWrapper（只有 forward）
    policy = PolicyWrapper(agent).to(device)
    policy.eval()

    return policy, obs_dim, act_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ckpt_xxxx.pt (PyTorch PPO checkpoint).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to save ONNX model.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    policy, obs_dim, act_dim = build_policy_from_checkpoint(
        args.checkpoint, device
    )

    # 导出时用的示例输入，值随便，shape 必须是 [1, obs_dim]
    dummy_input = torch.randn(1, obs_dim, device=device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[INFO] Exporting ONNX to: {args.output}")
    torch.onnx.export(
        policy,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        # 先用 legacy 导出就够了，dynamo=True 也行，先不折腾
        # dynamo=False  # 你也可以显式写上
    )
    print("[INFO] ONNX export done.")


if __name__ == "__main__":
    main()
