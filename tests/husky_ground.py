#!/usr/bin/env python
# coding: utf-8

"""
This script builds a ground from scratch.
"""

import sys
sys.path.append('..')

from worldcreator import USE_GPU
from worldcreator import representation
from worldcreator import factory
from worldcreator import converters

# define a projection of the scene

rasterization = representation.Rasterization(
    l_x=200, # size of the map in meter
    l_y=200, # size of the map in meter
    p_m=10, # resolution of the map: pixels per meter
)

# instantiate a ground builder

ground_builder = factory.HuskyGround(
    seed=19, # random seed
)

# generate the heightmap and the masks for the choosen projection

matrices = ground_builder.matrices(
    rasterization, # map projection
)

# export the matrices

for name, matrix in matrices.items():
    if USE_GPU:
        matrix = np.asnumpy(matrix)
    converters.ndarray_to_png(matrix, name, '../data/png/husky')
    converters.ndarray_to_npy(matrix, name, '../data/npy/husky')

# export the meshes corresponding to the ground zones

ground_builder.stls(
    rasterization, # map projection
    expstm="ground",
    expdir="../data/stl/husky",
    #hgtmap = ..., # to use your own heightmap
)
