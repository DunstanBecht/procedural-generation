#!/usr/bin/env python
# coding: utf-8

"""
This script builds a ground from a heightmap.
"""

import sys
sys.path.append('..')

import numpy as np
from worldcreator import representation
from worldcreator import factory
from worldcreator import converters

# define a projection of the scene

rasterization = representation.Rasterization(
    l_x=656, # size of the map
    l_y=656, # size of the map
    p_m=10, # resolution of the map
)

heightmap = np.load('../data/npy/lake/dem.npz')['data']
#heightmap = np.load('/home/gpu_user/Documents/isaac-custom-code/raw_generation/gen0/dem.npz')['data']
mask_forest = heightmap > 2
mask_lake = heightmap < 2

# export the matrices

converters.ndarray_to_png(heightmap, 'heightmap', '../data/png/lake')
converters.ndarray_to_npy(heightmap, 'heightmap', '../data/npy/lake')

converters.ndarray_to_png(mask_forest, 'forest', '../data/png/lake')
converters.ndarray_to_npy(mask_forest, 'forest', '../data/npy/lake')

converters.ndarray_to_png(mask_lake, 'lake', '../data/png/lake')
converters.ndarray_to_npy(mask_lake, 'lake', '../data/npy/lake')

# export the meshes corresponding to the ground zones

ground_builder = factory.GroundBuilder()

ground_builder.stls(
    rasterization, # map projection
    expstm="ground",
    expdir="../data/stl/lake",
    hgtmap = heightmap,
    zonsiz= 20, # zone size in meter
)
