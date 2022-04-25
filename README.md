# Procedural Environment Generation

This project was developed as part of the Special Problem CS8903 course at Georgia Institute of Technology.
The tools proposed here are intended to facilitate the procedural generation of worlds for use in robot training.

* Dunstan Becht <dunstan.becht@gatech.edu>
* Gaspard Burlin <gburlin3@gatech.edu>

## Python package installation

Install the following python packages:
* `numpy`
* `scipy`
* `cupy`
* `opencv-python`
* `pyfastnoisesimd`
* `vtk`
* `sklearn`
* `scikit-image`

## User guide

The main content of this repository is the **worldcreator** package.
You will find several examples of use in the scripts contained in directory `/tests`.
Start by looking at the script `/tests/representation.py`.
This one shows the general principles of vectorized objects, rasterization and embedding.
Then look at the script `/tests/husky_ground.py`.
Then look at the script `/tests/husky_forest.py`.
Then look at the script `/main1.py`.
Then look at the script `/main2.py`.

It is possible to use cupy instead of numpy.
To do so, you must modify the `USE_CPU` constant in `/worldcreator/__init__.py`.

## Useful links

Pixar USD:
* ![xformOp](https://graphics.pixar.com/usd/dev/api/xform_op_8h.html)
* ![GfRotation](https://graphics.pixar.com/usd/release/api/class_gf_rotation.html)
