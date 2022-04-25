#!/usr/bin/env python
# coding: utf-8

"""
This script loads a world from a JSON file and produces its USD file.
"""

import os
import omni
from omni.isaac.kit import SimulationApp
from worldcreator import factory
from worldcreator import representation

repo = "/home/gpu_user/.local/share/ov/pkg/isaac_sim-2021.2.0/standalone_examples/python_samples/CS8903"

world = "lake" # "lake"

# load the composed object representing the world

world_object = representation.load(f"{repo}/data/json/{world}/environment.json")

# start isaac sim

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}
simulation_app = SimulationApp(CONFIG)

from worldcreator.isaacutils import *

stage = omni.usd.get_context().get_stage()

# load the features in isaac

nucleus = get_nucleus_server()

world_object['trees'].prefix = os.path.join(nucleus, "NVIDIA/Assets/Vegetation/Trees")

put(world_object, stage)

# export the USD file

#omni.usd.get_context().save_as_stage(os.path.join(nucleus, f"world_{world}.usd"), None)

omni.usd.get_context().save_as_stage(f"{repo}/data/usd/{world}/world.usd", None)

simulation_app.close()
