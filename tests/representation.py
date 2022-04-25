#!/usr/bin/env python
# coding: utf-8

"""
This script tests the representation module.
"""

import sys
sys.path.append('..')

import matplotlib.pyplot as plt

from worldcreator import representation

# create two trees as atomic objects

tree_1 = representation.AtomicObject(
    name='tree_1', # name of the first tree
    usd='example.usd', # path to a USD file
    x=5, # x coordinate of the first tree
    y=4, # y coordinate of the first tree
    r=1, # radius of the tree
)

tree_2 = representation.AtomicObject(
    name='tree_2', # name of the first tree
    usd='example.usd', # path to a USD file
    x=5, # x coordinate of the first tree
    y=6, # y coordinate of the first tree
    r=1, # radius of the tree
)

# create a forest containing the two trees

forest = representation.ComposedObject(
    name="forest", # name of the forest
    components=[tree_1, tree_2], # objects contained in the forest
)

# define a projection of the scene

rasterization = representation.Rasterization(
    l_x=20, # size of the map
    l_y=10, # size of the map
    p_m=10, # resolution of the map
)

# display the first tree

print(tree_1.name)
mask = rasterization.mask(tree_1)
plt.imshow(mask)
plt.show()

# display the second tree

print(tree_2.name)
mask = rasterization.mask(tree_2)
plt.imshow(mask)
plt.show()

# display the forest

print(forest.name)
mask = rasterization.mask(forest)
plt.imshow(mask)
plt.show()

# display a simple gaussian function

print("a simple gaussian function")
gaussian = rasterization.gaussian(5, 5, 2)
plt.imshow(gaussian)
plt.show()

# display the forest attraction

print("probability of presence of an object attracted by the forest")
attraction = rasterization.attraction(forest)
plt.imshow(attraction)
plt.show()

# display the forest repulsion

print("probability of presence of an object repulsed by the forest")
repulsion = rasterization.repulsion(forest)
plt.imshow(repulsion)
plt.show()
