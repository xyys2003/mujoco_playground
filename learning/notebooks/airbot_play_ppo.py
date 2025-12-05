#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run PPO training for the Airbot Play pick task by reusing the generic
state-based PPO script.

This thin wrapper ensures the playground registry loads the Airbot Play
pick environment before invoking the shared training entrypoint.
"""

from mujoco_playground import registry

# Reuse all PPO plumbing from the state-based script.
from learning.notebooks import state_ppo as _ppo

# Force the environment globals to point to Airbot Play pick.
_ppo.env_name = "AirbotPlayPick"
_ppo.env = registry.load(_ppo.env_name)
_ppo.env_cfg = registry.get_default_config(_ppo.env_name)


if __name__ == "__main__":
    _ppo.main()
