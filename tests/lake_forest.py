#!/usr/bin/env python
# coding: utf-8

"""
This script builds a forest.
"""

import sys
sys.path.append('..')

import json
from worldcreator import *
from worldcreator import representation
from worldcreator import factory
from worldcreator import converters

# define a projection of the scene

rasterization = representation.Rasterization(
    l_x=656, # size of the map
    l_y=656, # size of the map
    p_m=10, # resolution of the map
)

# retrieve the heightmap and the forest mask

heightmap = np.load('../data/npy/lake/heightmap.npy')[0:6560, 0:6560]
forest_mask = np.load('../data/npy/lake/forest.npy')[0:6560, 0:6560]

# instantiate a forest builder

forest_builder = factory.ForestBuilder(
    d=0.01,
    n=100,
    seed=19,
    usds=[f'American_Beech.usd'],
)

# generate the forest in the choosen projection

trees, clusters = forest_builder.generate_fast(
    rasterization,
    forest_mask,
    heightmap,
)

# export a picture of the forest

m_trees = rasterization.mask(trees)
m_clusters = forest_mask*rasterization.attraction(clusters)

if USE_GPU:
    m_trees = np.asnumpy(m_trees)
    m_clusters = np.asnumpy(m_clusters)

converters.ndarray_to_png(m_trees, 'trees', '../data/png/lake')
converters.ndarray_to_png(m_clusters, 'clusters', '../data/png/lake')

# export the forest object

with open('../data/json/lake/forest.json', 'w') as f:
    json.dump(trees.json(), f)
