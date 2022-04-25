#!/usr/bin/env python
# coding: utf-8

"""
This script generates some USD files and save the world into a JSON file.
"""

import os
import json
import omni
from omni.isaac.kit import SimulationApp
from worldcreator import factory
from worldcreator import representation

repo = "/home/gpu_user/.local/share/ov/pkg/isaac_sim-2021.2.0/standalone_examples/python_samples/CS8903"

world = "lake" # "lake"

# convert the STL ground zones into USD files

if not os.path.exists(f"{repo}/data/usd/{world}/ground"):
    factory.GroundBuilder.usds(
        f"{repo}/data/stl/{world}/ground",
        f"{repo}/data/usd/{world}/ground",
    )

# create a composed object containing the ground zones

ground = factory.GroundBuilder.collect(f"{repo}/data/usd/{world}/ground")

# load the forest

trees = representation.load(f"{repo}/data/json/{world}/forest.json")

# create a sky 

sky = representation.AtomicObject(
    name='sky',
    usd='',
)

# assemble the features in a nex composed object

world_object = representation.ComposedObject(
    name='environment',
    components=[trees, ground],
)

# export the composed object representing the environment

with open(f'{repo}/data/json/{world}/environment.json', 'w') as f:
    json.dump(world_object.json(), f)
