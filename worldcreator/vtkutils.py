#!/usr/bin/env python
# coding: utf-8

"""
This module contains tools to handle meshes.
"""

import vtk
from . import np

def plane(
    o: tuple,
    n: tuple,
) -> vtk.vtkPolyData:
    """
    Return a plane.

    Input:
        o (tuple): origin of the plane
        n (tuple): normal of the plane

    Output:
        p (vtk.vtkPolyData): plane
    """
    p = vtk.vtkPlane()
    p.SetNormal(n)
    p.SetOrigin(o)
    return p

def cut(
    plane: vtk.vtkPolyData,
    poly: vtk.vtkPolyData,
) -> vtk.vtkPolyData:
    """
    Cut a VTK poly using a plane.

    Input:
        plane (vtk.vtkPolyData): plane of the cut
        poly (vtk.vtkPolyData): object to cut

    Output:
        vtkdat (vtk.vtkPolyData): mesh that aligns with the normal
    """
    cut = vtk.vtkClipPolyData()
    if vtk.VTK_MAJOR_VERSION <= 5:
        cut.SetInput(poly)
    else:
        cut.SetInputData(poly)
    cut.SetClipFunction(plane)
    cut.Update()
    return cut.GetOutput()

def clear_offset(
    poly: vtk.vtkPolyData,
) -> vtk.vtkPolyData:
    """
    Set the origin of the mesh to 0,0,0.

    Input:
        poly (vtk.vtkPolyData): mesh to translate

    Output:
        vtkdat (vtk.vtkPolyData): translated mesh
    """
    bounds = poly.GetBounds()
    transform = vtk.vtkTransform()
    transform.Translate(-bounds[0], -bounds[2], 0)
    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetTransform(transform)
    filter.SetInputData(poly)
    filter.Update()
    return filter.GetOutput()

def makeFlatSurface(
    px,
    py,
    z,
    size,
):
    """
    Creates flat tiles.
    """
    #Array of vectors containing the coordinates of each point
    nodes = np.array([[px, py, z], [px + size[0]//2, py, z], [px+size[0], py, z],
                      [px+size[0], py+size[1]//2, z], [px+size[0], py+size[1], z],
                      [px+size[0]//2, py+size[1], z], [px, py+size[1], z], [px, py+size[1]//2, z],
                      [px+size[0]//2, py+size[1]//2, z]])
    #Array of tuples containing the nodes correspondent of each element
    elements =[(0, 1, 8, 7), (7, 8, 5, 6), (1, 2, 3, 8), (8, 3, 4,
                        5)]
    #Make the building blocks of polyData attributes
    Mesh = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()
    #Load the point and cell's attributes
    for i in range(len(nodes)):
        Points.InsertPoint(i, nodes[i])
    for i in range(len(elements)):
        Cells.InsertNextCell(mkVtkIdList(elements[i]))
    #Assign pieces to vtkPolyData
    Mesh.SetPoints(Points)
    Mesh.SetPolys(Cells)
    return Mesh

def mkVtkIdList(
    it,
):
    """
    Creates a VTK Id list.
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil
