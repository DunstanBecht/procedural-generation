#!/usr/bin/env python
# coding: utf-8

"""
This module contains tools to switch from one format to another.
"""

import os
import asyncio
from PIL import Image
import vtk
import numpy
from vtk.util import numpy_support
from . import np
from . import vtkutils

def ndarray_to_png(
    matrix: numpy.ndarray,
    expstm: str = "map",
    expdir: str = "",
) -> None:
    """
    Save a map to an image file.

    Input:
        matrix (ndarray): 2D matrix to be exported
        expstm (str): name of the exported file without the suffix
        expdir (str): path to the folder where to export the file
    """
    path = os.path.join(expdir, f"{expstm}.png")
    if matrix.dtype==bool:
        matrix = matrix.astype(int)
    if len(matrix.shape)>2 and matrix.shape[2]>1:
        mode = 'RGB'
        if matrix.dtype != np.uint8:
            raise Exception('not 8 bit data array')
    else:
        mode = 'L'
        matrix = numpy.rint((matrix-matrix.min())/matrix.ptp()*255)
        matrix = matrix.astype(np.uint8)
    img = Image.fromarray(matrix, mode)
    img.save(path)
    # img.show()

def ndarray_to_npy(
    matrix: np.ndarray,
    expstm: str = "map",
    expdir: str = "",
) -> None:
    """
    Save a map to a numpy file.

    Input:
        matrix (ndarray): 2D matrix to be exported
        expstm (str): name of the exported file without the suffix
        expdir (str): path to the folder where to export the file
    """
    path = os.path.join(expdir, f"{expstm}.npy")
    np.save(path, matrix)

def ndarray_to_vtkPolyData(
    hgtmap: np.ndarray,
    abserr: float = 1.0,
    scales: tuple = (1, 1, 1),
) -> vtk.vtkPolyData:
    """
    Return the mesh correspoding to the heightmap.

    Input:
        hgtmap (ndarray): height map
        abserr (float): error in height
        scales (tuple): scale factor in the three directions

    Output:
        vtkdat (vtkPolyData): mesh
    """
    hgtmap = hgtmap.T
    row_size, col_size = hgtmap.shape
    number_of_elevation_entries = hgtmap.size
    vectorized_elevations = np.reshape(
        hgtmap,
        (number_of_elevation_entries, 1),
    )
    vtk_array = numpy_support.numpy_to_vtk(
        vectorized_elevations,
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )
    # make a VTK heightmap
    image = vtk.vtkImageData()
    image.SetDimensions(col_size, row_size, 1)
    image.AllocateScalars(vtk_array.GetDataType(), 4)
    image.GetPointData().GetScalars().DeepCopy(vtk_array)
    # decimate the heightmap
    deci = vtk.vtkGreedyTerrainDecimation()
    deci.SetInputData(image)
    deci.BoundaryVertexDeletionOn()
    deci.SetErrorMeasureToAbsoluteError()
    deci.SetAbsoluteError(abserr)
    deci.Update()
    tvolume = deci.GetOutput()
    # set the scale correctly
    transform = vtk.vtkTransform()
    transform.Scale(scales)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(tvolume)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    tvolume = transformFilter.GetOutput()
    # return scaled object
    return tvolume

def vtkPolyData_to_stl(
    vtkdat: vtk.vtkPolyData,
    expstm: str = "mesh",
    expdir: str = "",
) -> None:
    """
    Save the VTK poly as an STL.

    Input:
        vtkdat (vtkPolyData): mesh
        expstm (str): name of the exported file without the suffix
        expdir (str): path to the folder where to export the file
    """
    path = os.path.join(expdir, f"{expstm}.stl")
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(path)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(vtkdat)
    else:
        writer.SetInputData(vtkdat)
    writer.Write()

def vtkPolyData_to_stl_parts(
    vtkdat: vtk.vtkPolyData,
    expstm: str = "mesh_parts",
    expdir: str = "",
    size: int = 50,
    poly_max_x: int = 750,
    poly_max_y: int = 750,
) -> None:
    """
    Cut the mesh into a smaller meshes of a given size.

    Input:
        vtkdat (vtkPolyData): mesh
        expstm (str): name of the exported folder
        expdir (str): path to the folder where to export the result
        size (int): size of the subareas
        poly_max_x (int): size of the mesh
        poly_max_y (int): size of the mesh

    The way this works in by creating planes and cutting along these planes.
    """
    path = os.path.join(expdir, expstm)
    if not os.path.isdir(path):
        os.makedirs(path)
    for i in range(size, poly_max_x+size, size):
        # cut and get left side of the mesh
        plane = vtkutils.plane((0, i, 0), (0, -1, 0))
        tmp = vtkutils.cut(plane, vtkdat)
        # we just got a strip that we are going to cut into cubes.
        for j in range(size, poly_max_y+size, size):
            plane = vtkutils.plane((j*1.0, 0, 0), (-1, 0, 0))
            output = vtkutils.cut(plane, tmp)
            plane = vtkutils.plane((j*1.0, 0, 0), (1, 0, 0))
            tmp = vtkutils.cut(plane, tmp)
            Bounds = output.GetBounds()
            if np.abs(Bounds[-1] - Bounds[-2]) < 0.15:
                output = vtkutils.makeFlatSurface(
                    j-size,
                    i-size,
                    np.mean([Bounds[-1], Bounds[-2]]),
                    (np.min([Bounds[1]-Bounds[0], size]),
                     np.min([Bounds[3]-Bounds[2], size])),
                )
            output = vtkutils.clear_offset(output)
            expstm = f"{j-size}_{i-size}" # <x coordinate>_<y coordinate>
            vtkPolyData_to_stl(output, expdir=path, expstm=expstm)
        # cut and get right side of the mesh
        plane = vtkutils.plane((0, i, 0), (0, 1, 0))
        vtkdat = vtkutils.cut(plane, vtkdat)

async def stl_to_usd(
    in_file: str,
    out_file: str,
    load_materials: bool = False,
) -> bool:
    """
    Convert an STL file into a USD file.

    Input:
        in_file (str): path to the STL file
        out_file (str): path to the USD file

    Output:
        success (bool): success of the operation
    """
    import omni.kit.asset_converter
    def progress_callback(progress, total_steps):
        pass
    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    converter_context.ignore_cameras = True
    converter_context.single_mesh = True
    converter_context.use_meter_as_world_unit = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file,
        out_file,
        progress_callback,
        converter_context,
    )
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success

def stl_parts_to_usd_parts(
    impstm: str,
    impdir: str = "",
    expstm: str = "scene_parts",
    expdir: str = "",
) -> None:
    """
    Convert multiple stl files into usd files.

    Imput:
        impstm (str): name of the imported folder containing the stl files
        impdir (str): path to the folder where to import the input folder
        expstm (str): name of the exported folder containing the usd files
        expdir (str): path to the folder where to export the result folder
    """
    folder_in = os.path.join(impdir, impstm)
    folder_out = os.path.join(expdir, expstm)

    print(f"\nConverting folder {folder_in}...")
    models = os.listdir(folder_in)
    models.sort()
    for model in models:
        imppth = os.path.join(folder_in, model)
        stem, format = os.path.splitext(model)
        if format in [".stl", ".obj", ".fbx"]:
            exppth = os.path.join(folder_out, f"{stem}.usd")
            status = asyncio.get_event_loop().run_until_complete(stl_to_usd(imppth, exppth, False))
            if not status:
                print(f"ERROR Status is {status}")
            print(f"---Added {exppth}")
