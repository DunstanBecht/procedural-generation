#!/usr/bin/env python
# coding: utf-8

"""
This module contains tools to handle objects in isaac sim.
"""

import os
from os import cpu_count
import omni
import carb
from numpy import sin, cos
import pxr
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdShade
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.physx.scripts import utils
from .representation import *

def put(
    o,#AtomicObject|ComposedObject,
    s: pxr.Usd.Stage,
    p: str = "",
) -> None:
    """
    Put object(s) on scene.

    Input:
        o (AtomicObject|ComposedObject) object
        s (): stage
        p (str): supplementary path prefix
    """
    def rec(a, n='', p2=''):
        if isinstance(a, AtomicObject):
            print(a.name, (a.x, a.y, a.z))
            createObject(
                n+'/'+a.name,
                s,
                os.path.join(p, p2, a.prefix, a.usd),
                False,
                position=Gf.Vec3d(a.x*100, a.y*100, a.z*100),
            )
        if isinstance(a, ComposedObject):
            for b in a:
                rec(b, n+"/"+a.name, a.prefix)
    rec(o)

def CreateBasicMaterial(
    stage,
):
    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created_list,
    )
    mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
    material = UsdShade.Material(mtl_prim)
    return material

def createObject(
    prefix,
    stage,
    path,
    material,
    position=Gf.Vec3d(0, 0, 0),
    rotation=Gf.Rotation(Gf.Vec3d(0,0,1), 0),
    group=[],
    allow_physics=False,
    density=1000,
    scale=Gf.Vec3d(1.0,1.0,1.0),
    is_instance=True,
    collision=False,
):
    """
    Creates a 3D object from a USD file and adds it to the stage.
    """
    prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
    group.append(prim_path)
    obj_prim = stage.DefinePrim(prim_path, "Xform")
    obj_prim.GetReferences().AddReference(path)
    if is_instance:
        obj_prim.SetInstanceable(True)
    xform = UsdGeom.Xformable(obj_prim)

    set_scale(xform, scale)
    set_transform(xform, get_transform(rotation, position))

    if material:
        UsdShade.MaterialBindingAPI(obj_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

    if allow_physics:
        utils.setRigidBody(obj_prim, "convexHull", False)
        mass_api = UsdPhysics.MassAPI.Apply(obj_prim)
        mass_api.CreateMassAttr(density)
    if collision:
        utils.setCollider(obj_prim, approximationShape="convexHull")
    return group

def get_transform(
    rotation: Gf.Rotation,
    position: Gf.Vec3d,
) -> Gf.Matrix4d:
    matrix_4d = Gf.Matrix4d().SetTranslate(position)
    matrix_4d.SetRotateOnly(rotation)
    return matrix_4d

def set_property(
    xform: UsdGeom.Xformable,
    value,
    property,
) -> None:
    op = None
    for xformOp in xform.GetOrderedXformOps():
        if xformOp.GetOpType() == property:
            op = xformOp
    if op:
        xform_op = op
    else:
        xform_op = xform.AddXformOp(
            property,
            UsdGeom.XformOp.PrecisionDouble,
            "",
        )
    xform_op.Set(value)

def set_scale(
    xform: UsdGeom.Xformable,
    value,
) -> None:
    set_property(xform, value, UsdGeom.XformOp.TypeScale)

def set_translate(
    xform: UsdGeom.Xformable,
    value,
) -> None:
    set_property(xform, value, UsdGeom.XformOp.TypeTranslate)

def set_rotate_xyz(
    xform: UsdGeom.Xformable,
    value,
) -> None:
    set_property(xform, value, UsdGeom.XformOp.TypeRotateXYZ)

def set_transform(
    xform: UsdGeom.Xformable,
    value: Gf.Matrix4d,
) -> None:
    set_property(xform, value, UsdGeom.XformOp.TypeTransform)

def create_ground_plane(
    stage,
    plane_name,
    size=10,
    up_direction="Z",
    location=Gf.Vec3f(0,0,0),
    unknown=Gf.Vec3f(1.0),
    visible=True,
):
    from pxr import PhysicsSchemaTools, PhysxSchema
    """
    Creates a ground plane.
    Inputs:
        stage: The name of the world the objects belong to.
        plane_name: The name of the ground plane.
        size: The size of the ground plane in meters.
        up_direction: The direction normal to the plane.
        location: The position of the plane.
        unknown: An unknown parameter.
        visible: Should the ground plane be visible.
    """
    PhysicsSchemaTools.addGroundPlane(stage, plane_name, up_direction, size*10, location, unknown)
    if not visible:
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(plane_name))
        if imageable:
            imageable.MakeInvisible()

def setup_cpu_physics(
    stage,
    physics_name,
    gravity=9.81,
    gravity_direction=Gf.Vec3f(0.0, 0.0, -1.0),
):
    from pxr import PhysicsSchemaTools, PhysxSchema
    # Add physics scene
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(physics_name))
    # Set gravity vector
    scene.CreateGravityDirectionAttr().Set(gravity_direction)
    scene.CreateGravityMagnitudeAttr().Set(gravity*100)
    # Set physics scene to use cpu physics
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_name))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage,physics_name)
    physxSceneAPI.CreateEnableCCDAttr(True)
    physxSceneAPI.CreateEnableStabilizationAttr(True)
    physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreateSolverTypeAttr("TGS")

def get_nucleus_server():
    result, nucleus_server = find_nucleus_server()
    if result is False:
        carb.log_error(
            "Could not find nucleus server. Stopping."
        )
        exit(1)
    return nucleus_server
