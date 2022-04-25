#!/usr/bin/env python
# coding: utf-8

"""
This module facilitates the representation of an environment.
"""

import json
import skimage.draw
import numpy
from . import np, USE_GPU

# class declaration

class AtomicObject(object):
    pass

class ComposedObject(object):
    pass

class Rasterization(object):
    pass

# class implementation

class AtomicObject(object):
    """
    This class represents an object that does not contain sub-parts.
    """

    __slots__ = ('name', 'usd', 'x', 'y', 'z', 'r', 'prefix')

    def __init__(self, **kwargs):
        """
        Instantiate an atomic object.

        Input:
          **name (str): name of the object
          **usd (str): path to the usd file
          **x (float|int): x coordinate in meter
          **y (float|int): y coordinate in meter
          **z (float|int): z coordinate in meter
          **r (float|int): radius of the object in meter
        """
        self.name = kwargs.get('name', "untitled atomic object")
        self.usd = kwargs.get('usd', None)
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.z = kwargs.get('z', 0)
        self.r = kwargs.get('r', 1)
        self.prefix = kwargs.get('prefix', '')

    def json(self):
        """
        Return the json representation of the object.

        Output:
            d (dict): json data
        """
        d = {'type': 'AtomicObject'}
        for a in self.__slots__:
            d[a] = getattr(self, a)
        return d

class ComposedObject(object):
    """
    This class represents an object composed of sub-objects.
    """

    __slots__ = ('name', 'components', 'prefix')

    def __init__(self, **kwargs):
        """
        Instantiate a composed object.

        Input:
          **name (str): name of the object
          **components (list): sub objects list
        """
        self.name = kwargs.get('name', "untitled composed object")
        self.components = kwargs.get('components', [])
        self.prefix = kwargs.get('prefix', '')

    def __iter__(self):
        for o in self.components:
            yield o

    def __getitem__(self, 
        n: str,
    ):
        """
        Get a component by its name.

        Input:
            n (str): name of the component

        Output:
            o (AtomicObject|ComposedObject): component
        """
        for obj in self:
            if isinstance(obj, (ComposedObject, AtomicObject)):
                if obj.name == n:
                    return obj
        raise KeyError()

    def append(self,
        o, #: AtomicObject|ComposedObject
    ) -> None:
        """
        Add a sub-object.

        Input:
            o (AtomicObject|ComposedObject): object to add
        """
        self.components.append(o)

    def json(self):
        """
        Return the json representation of the object.

        Output:
            d (dict): json data
        """
        d = {
            'type': 'ComposedObject',
            'name': self.name,
            'components': [o.json() for o in self.components],
            'prefix': self.prefix,
        }
        return d

def evaluate(
    d: dict,
):
    """
    """
    if d['type'] == "AtomicObject":
        return AtomicObject(**d)
    elif d['type'] == "ComposedObject":
        d['components'] = [evaluate(c) for c in d['components']]
        return ComposedObject(**d)
    else:
        raise Exception()


def load(
    path: str,
):
    """

    """
    with open(path, 'r') as f:
        data = json.load(f)
    return evaluate(data)

def normalize(
    a: np.ndarray,
) -> np.ndarray:
    """
    Normalize and return an array of probabilities.

    Input:
        a (np.ndarray): array to normalize
        b (np.ndarray): normalized array
    """
    a -= a.min()
    s = np.sum(a)
    if s == 0:
        return a + 1/a.shape[0]*a.shape[1]
    return a/s

class Rasterization(object):
    """
    This class allows the transformation of vector objects to a matrix object.
    """

    def __init__(self, **kwargs):
        """
        Instantiate a rasterization object.

        Input:
          **l_x (float|int): map size in x direction in meter
          **l_y (float|int): map size in y direction in meter
          **p_m (float|int): number of pixels per meter
          **o_x (float|int): x coordinate of the origin in meter
          **o_y (float|int): y coordinate of the origin in meter
        """
        self.l_x = kwargs.get('l_x')
        self.l_y = kwargs.get('l_y')
        self.p_m = kwargs.get('p_m')
        self.o_x = kwargs.get('o_x', 0)
        self.o_y = kwargs.get('o_y', 0)
        self.shape = (round(self.l_x*self.p_m), round(self.l_y*self.p_m))

    def indexes(self,
        x, #: float|int,
        y, #: float|int,
    ) -> tuple:
        """
        Return the indexes corresponding to the position in the matrix.

        Input:
            x (float|int): x coordinate in meter
            y (float|int): y coordinate in meter

        Output:
            i_x (int): x coordinate index in the 2D array
            i_y (int): y coordinate index in the 2D array
        """
        i_x = np.round((x-self.o_x)*self.p_m)
        i_y = np.round((y-self.o_y)*self.p_m)
        return int(i_x), int(i_y)

    def position(self,
        i_x: int,
        i_y: int,
    ) -> tuple:
        """
        Return the position in meter corresponding to the indexes.

        Input:
            i_x (int): x coordinate index in the 2D array
            i_y (int): y coordinate index in the 2D array

        Output:
            x (float|int): x coordinate in meter
            y (float|int): y coordinate in meter
        """
        x = i_x/self.p_m + self.o_x
        y = i_y/self.p_m + self.o_y
        return float(x), float(y)

    def mask(self,
        o, #: AtomicObject|ComposedObject,
    ) -> np.ndarray:
        """
        Return the space occupied by the object.

        Input:
            o (AtomicObject|ComposedObject): vector object

        Output:
            m (np.ndarray): mask
        """
        canvas = numpy.zeros(self.shape, dtype=numpy.int8)
        def rec(a):
            if isinstance(a, AtomicObject):
                p = self.indexes(a.x, a.y)
                r = a.r * self.p_m
                m = skimage.draw.disk(p, radius=r, shape=canvas.shape)
                canvas[m] = 1
            if isinstance(a, ComposedObject):
                for b in a:
                    rec(b)
        rec(o)
        return np.array(canvas)

    def gaussian(self,
        x, #: float|int,
        y, #: float|int,
        s, #: float|int,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Return a 2D Gaussian function.

        Input:
            x (float|int): x coordinate of the peak in meter
            y (float|int): y coordinate of the peak in meter
            s (float|int): standard deviation in meter
            normalize (bool): the function is normalized

        Output:
            g (np.ndarray): gaussian function array
        """
        i_x, i_y = self.indexes(x, y)
        i_s = s * self.p_m
        ax_x = np.arange(self.shape[0])
        ax_y = np.arange(self.shape[1])
        exp_x = np.exp(-((ax_x-i_x)**2/i_s**2)/2)
        exp_y = np.exp(-((ax_y-i_y)**2/i_s**2)/2)
        canvas = np.outer(exp_x, exp_y)
        if normalize:
            canvas = normalize(canvas)
        return canvas

    def attraction(self,
        o, #: AtomicObject|ComposedObject,
    ) -> np.ndarray:
        """
        Return the probability of presence of an attracted object.

        Input:
            o (AtomicObject|ComposedObject): vector object

        Output:
            p (np.ndarray): array of probabilities
        """
        def rec(a):
            if isinstance(a, AtomicObject):
                return self.gaussian(a.x, a.y, a.r)
            if isinstance(a, ComposedObject):
                canvas = np.zeros(self.shape)
                for b in a:
                    canvas += rec(b)
                return canvas
        p = rec(o)
        return normalize(p)

    def repulsion(self,
        o, #: AtomicObject|ComposedObject,
    ) -> np.ndarray:
        """
        Return the probability of presence of a repulsed object.

        Input:
            o (AtomicObject|ComposedObject): vector object

        Output:
            p (np.ndarray): array of probabilities
        """
        return normalize(1 - self.attraction(o))
