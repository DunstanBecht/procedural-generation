#!/usr/bin/env python
# coding: utf-8

"""
This package contains tools to create worlds.
"""

__author__ = "Dunstan Becht"
__version__ = "0.8.0"

USE_GPU = False

if USE_GPU:
    import cupy as np
    #np.cuda.Device(0)
    def unravel_index(*args, **kwargs):
        kwargs['dims'] = kwargs.pop('shape', None)
        return np.unravel_index(*args, **kwargs)
else:
    import numpy as np
    unravel_index = np.unravel_index
