#!/usr/bin/env python
# coding: utf-8

"""
This module contains tools for building certain environments.
"""

import os
import math
import vtk
import pyfastnoisesimd as fns
import cv2
from . import np, unravel_index
from . import representation
from . import converters

def random_index(
    p: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    """
    Return a random position.

    Input:
        p (np.ndarray): probabilities
        n (int): number of points

    Output:
        i (np.ndarray): random position
    """
    r = np.ravel(p)
    indexes = np.array(unravel_index(
        np.random.choice(np.arange(len(r)), size=n, p=r, replace=False),
        shape=p.shape,
    ))
    if n==1:
        return indexes
    return indexes.T

class Builder(object):
    """
    This class represents a builder.
    """

    def __init__(self, **kwargs):
        """
        Instantiate a builder.

        Input:
          **seed (int): random seed
        """
        self.seed = kwargs.get('seed', 0)

class ForestBuilder(Builder):
    """
    This class represents a forest builder.
    """

    def __init__(self, **kwargs):
        """
        Instantiate a forest builder.

        Input:
          **d (float|int): tree density [tree/meter^2]
          **n (float|int): mean number of trees per cluster
          **r (float): radius of an object in meter
          **usds (list): list of path to the usd files
        """
        super().__init__(**kwargs)
        self.d = kwargs.get('d', 0.01)
        self.n = kwargs.get('n', 100)
        self.r = kwargs.get('r', 1)
        self.usds = kwargs.get('usds', ['example.usd'])

    def generate(self,
        rasterization: representation.Rasterization,
        m: np.ndarray = None,
        h: np.ndarray = None,
    ) -> representation.ComposedObject:
        """
        Generate and return a forest for a small map.

        Input:
            rasterization (representation.Rasterization): rasterization object
            m (np.ndarray): mask delimiting the forest area
            h (np.ndarray): height map

        Output:
            f (representation.ComposedObject): forest
        """
        np.random.seed(self.seed)
        if m is None:
            m = np.ones(rasterization.shape)
        else:
            m = m[0:rasterization.shape[0], 0:rasterization.shape[1]]
            if m.shape != rasterization.shape:
                raise ValueError("invalid mask shape")
        if h is None:
            h = np.zeros(rasterization.shape)
        s_p = np.sum(np.where(m, 1, 0)) # surface in pixels
        s_m = s_p / rasterization.p_m**2 # surface in meters
        trees = representation.ComposedObject(name='trees') # trees container
        clusters = representation.ComposedObject(name='clusters') # clusters container
        n_trees = int(math.ceil(s_m*self.d)) # number of trees
        n_cluster = int(math.ceil(n_trees/self.n)) # number of clusters
        repulsion_trees = m/s_p
        repulsion_clusters = m/s_p
        print(f"generating {n_trees} trees on {n_cluster} clusters")
        # place clusters
        for i in range(1, n_cluster+1):
            print(f'cluster {i}')
            indexes = random_index(repulsion_clusters)
            x, y = rasterization.position(*indexes)
            c = representation.AtomicObject(
                name=f'cluster_{i}',
                x=x,
                y=y,
                r=20,
            )
            repulsion_c = representation.normalize(np.where(m, rasterization.repulsion(c), 0))
            repulsion_clusters = (repulsion_clusters*(i-1) + repulsion_c)/i
            clusters.append(c)
        attraction_clusters = rasterization.attraction(clusters)
        attraction_clusters = representation.normalize(np.where(m, attraction_clusters, 0))
        p = attraction_clusters
        # place trees
        for i in range(1, n_trees+1):
            print(f'tree {i}')
            indexes = tuple(random_index(p))
            x, y = rasterization.position(*indexes)
            z = float(h[indexes]/rasterization.p_m)
            t = representation.AtomicObject(
                name=f'tree_{i}',
                x=x,
                y=y,
                z=z,
                r=1,
                #usd=self.usds[rng.integers(len(self.usds))],
            )
            repulsion_t = representation.normalize(np.where(m, rasterization.repulsion(t), 0))
            repulsion_trees = (repulsion_trees*(i-1) + repulsion_t)/i
            p = representation.normalize(attraction_clusters * repulsion_trees)
            trees.append(t)
        return representation.ComposedObject(name='forest', components=[trees, clusters])

    def generate_fast(self,
        rasterization: representation.Rasterization,
        m: np.ndarray = None,
        h: np.ndarray = None,
    ) -> representation.ComposedObject:
        """
        Generate and return a forest for a big map.

        Input:
            rasterization (representation.Rasterization): rasterization object
            m (np.ndarray): mask delimiting the forest area
            h (np.ndarray): height map

        Output:
            f (representation.ComposedObject): forest
        """
        np.random.seed(self.seed)
        if m is None:
            m = np.ones(rasterization.shape)
        else:
            m = m[0:rasterization.shape[0], 0:rasterization.shape[1]]
            if m.shape != rasterization.shape:
                raise ValueError("invalid mask shape")
        if h is None:
            h = np.zeros(rasterization.shape)
        s_p = np.sum(np.where(m, 1, 0)) # surface in pixels
        s_m = s_p / rasterization.p_m**2 # surface in meters
        trees = representation.ComposedObject(name='trees') # container
        clusters = representation.ComposedObject(name='clusters') # container
        n_trees = int(math.ceil(s_m*self.d)) # number of trees
        n_cluster = int(math.ceil(n_trees/self.n)) # number of clusters
        repulsion_clusters = m/s_p
        r_p = self.r*rasterization.p_m
        box_size = 4*r_p
        print(f"generating {n_trees} trees on {n_cluster} clusters")
        # place clusters
        for i in range(1, n_cluster+1):
            print(f'cluster {i}')
            indexes = random_index(repulsion_clusters)
            x, y = rasterization.position(*indexes)
            c = representation.AtomicObject(
                name=f'cluster_{i}',
                x=x,
                y=y,
                r=20,
            )
            repulsion_c = representation.normalize(np.where(m, rasterization.repulsion(c), 0))
            repulsion_clusters = (repulsion_clusters*(i-1) + repulsion_c)/i
            clusters.append(c)
        attraction_clusters = rasterization.attraction(clusters)
        attraction_clusters = representation.normalize(np.where(m, attraction_clusters, 0))
        p = attraction_clusters
        # place trees
        p_low_res = p[box_size//2::box_size, box_size//2::box_size]
        p_low_res = representation.normalize(p_low_res)
        indexes = random_index(p_low_res, n=n_trees)*box_size
        indexes += np.random.randint(-r_p, r_p, size=(n_trees, 2))
        for i in range(1, n_trees+1):
            print(f'tree {i}')
            ij = tuple(indexes[i-1])
            x, y = rasterization.position(*ij)
            z = float(h[ij]/rasterization.p_m)
            t = representation.AtomicObject(
                name=f'tree_{i}',
                x=x,
                y=y,
                z=z,
                r=self.r,
                usd=self.usds[np.random.randint(len(self.usds))],
            )
            trees.append(t)
        return representation.ComposedObject(name='forest', components=[trees, clusters])

class GroundBuilder(Builder):
    """
    This class represents a ground builder.
    """

    def usds(
        stldir: str,
        usddir: str,
    ) -> None:
        """
        Convert the STL ground zones to USD files.

        Input:
            stldir (str): path to the directory containing the stl files
            usddir (str): path to the directory containing the usd files
        """

        CONFIG = {
            "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
            "renderer": "RayTracedLighting",
            "headless": True,
        }
        from omni.isaac.kit import SimulationApp
        simulation_app = SimulationApp(CONFIG)

        converters.stl_parts_to_usd_parts(
            stldir,
            expstm=usddir,
        )

        simulation_app.close()

    def collect(
        usddir: str,
    ) -> representation.ComposedObject:
        """
        Load and return the ground from a directory.

        Input:
            usddir (str): path to the directory containing the USD files

        Output:
            ground (representation.ComposedObject): ground
        """
        ground = representation.ComposedObject(
            name='ground',
            prefix=usddir,
        )
        for entry in os.listdir(usddir):
            stem = entry.split(".")[0]
            x, y = stem.split("_")
            t = representation.AtomicObject(
                name=f'zone_{stem}',
                usd=entry,
                x=int(x)/100,
                y=int(y)/100,
            )
            ground.append(t)
        return ground

    def __init__(self, **kwargs):
        """
        Instantiate a ground builder.
        """
        super().__init__(**kwargs)

    def mesh(self,
        rasterization: representation.Rasterization,
      **kwargs,
    ) -> vtk.vtkPolyData:
        """
        Return the mesh corresponding to the ground.

        Input:
            rasterization (representation.Rasterization): rasterization object
          **hgtmap (ndarray): height map

        Output:
            vtkdat (vtkPolyData): mesh
        """
        hgtmap = kwargs.get('hgtmap', self.matrices(rasterization)['heightmap'])
        vtkdat = converters.ndarray_to_vtkPolyData(
            hgtmap,
            scales=tuple(3*[100/rasterization.p_m]),
            abserr=1, # the resolution is the same as rasterization.p_m
        )
        return vtkdat

    def stls(self,
        rasterization: representation.Rasterization,
      **kwargs,
    ) -> None:
        """
        Save the ground zones into stl files.

        Input:
            rasterization (representation.Rasterization): rasterization object
          **hgtmap (ndarray): height map
          **vtkdat (vtkPolyData): mesh
          **expdir (str): exportation directory
          **zonsiz (int): size of a zone in meter
        """
        vtkdat = kwargs.get('vtkdat', self.mesh(rasterization, **kwargs))
        expstm = kwargs.get('expstm', 'ground_zones')
        expdir = kwargs.get('expdir', '')
        zonsiz = kwargs.get('zonsiz', 10)
        converters.vtkPolyData_to_stl_parts(
            vtkdat,
            expstm=expstm,
            expdir=expdir,
            size=zonsiz*100,
            poly_max_x=rasterization.l_x*100, # useless
            poly_max_y=rasterization.l_y*100,
        )

    def matrices(self,
        rasterization: representation.Rasterization,
    ) -> dict:
        """
        Return the height map and the lake mask.

        Input:
            rasterization (representation.Rasterization): rasterization object

        Output:
            d (dict): heightmap and masks

        The height is given in meter.
        """
        d = {
            'heightmap': np.zeros(rasterization.shape),
            'forest': np.ones(rasterization.shape),
        }
        return d

class HuskyGround(GroundBuilder):
    """
    This class represents a ground builder for Husky.
    """

    def matrices(self,
        rasterization: representation.Rasterization,
    ) -> dict:
        """
        Return the height map and the lake mask.

        Input:
            rasterization (representation.Rasterization): rasterization object

        Output:
            d (dict): heightmap and masks

        The height is given in meter.
        """
        shape = rasterization.shape

        N_threads = 4

        perlin0 = fns.Noise(seed=self.seed, numWorkers=N_threads)
        perlin0.frequency = 1 / 100
        perlin0.noiseType = fns.NoiseType.Perlin
        perlin0.fractal.octaves = 16
        perlin0.fractal.lacunarity = 2.0
        perlin0.fractal.gain = 0.45
        perlin0.perturb.perturbType = fns.PerturbType.NoPerturb

        perlin01 = fns.Noise(seed=self.seed, numWorkers=N_threads)
        perlin01.frequency = 1 / 100
        perlin01.noiseType = fns.NoiseType.Perlin
        perlin01.fractal.octaves = 16
        perlin01.fractal.lacunarity = 2.0
        perlin01.fractal.gain = 0.45
        perlin01.perturb.perturbType = fns.PerturbType.NoPerturb

        shape1 = [60, 60]
        shape2 = [120, 120]
        m0 = perlin0.genAsGrid(shape1)
        m01 = perlin01.genAsGrid(shape2)

        m1 = cv2.resize(m0, shape) + cv2.resize(m01, shape)

        p = np.percentile(m1, 30)
        m1 -= p
        m1 *= 40

        m1label = (m1 < 0).astype(np.int8)
        labels = cv2.connectedComponents(m1label)

        d = {
            'heightmap': m1,
            'forest': labels[1]==0,
            'lake': labels[1]!=0,
        }
        return d
