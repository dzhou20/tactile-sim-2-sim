# unit test for tactile sensing (simulation)

import os
import sys
import argparse
import json
import math
import time
import csv
import random
import subprocess
import atexit
import signal

# 不生成 __pycache__
sys.dont_write_bytecode = True

# Isaac Sim 4.5.0: use isaacsim.simulation_app (not omni.isaac.kit)
from isaacsim.simulation_app import SimulationApp


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_get_repo_root(), path)


def _get_usd_path(usd_path: str | None = None) -> str:
    repo_root = _get_repo_root()
    default_path = os.path.join(repo_root, "assets", "isaac_sim", "ray_basic_test.usd")
    if not usd_path:
        return default_path
    if os.path.isabs(usd_path):
        return usd_path

    cwd_path = os.path.abspath(usd_path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(repo_root, usd_path)

def _ensure_big_cube(stage, cube_path="/World/BigRayCube", size=0.2):
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(cube_path)
    if prim and prim.IsValid():
        return prim
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.CreateSizeAttr(size)
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0.0, 0.0, size * 0.5))
    return cube.GetPrim()


def _ensure_big_sphere(stage, sphere_path="/World/BigSphere", radius=0.15, cube_path="/World/BigRayCube", padding=0.02):
    from pxr import UsdGeom, Gf, Usd
    prim = stage.GetPrimAtPath(sphere_path)
    if prim and prim.IsValid():
        return prim
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.GetRadiusAttr().Set(radius)
    # Place along the ray path: in front of the cube on the +X side
    cube_prim = stage.GetPrimAtPath(cube_path)
    if cube_prim and cube_prim.IsValid():
        xformable = UsdGeom.Xformable(cube_prim)
        xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        center = xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        cube = UsdGeom.Cube(cube_prim)
        size = float(cube.GetSizeAttr().Get() or 0.2)
        half = size * 0.5
        x = center[0] + (half + padding + radius)
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(x, center[1], center[2]))
    else:
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(-0.3, 0.0, radius))
    return sphere.GetPrim()


def _apply_rigidbody(
    prim,
    mass=1.0,
    enable_gravity=True,
    kinematic=False,
    collision_approximation: str | None = None,
):
    from pxr import UsdGeom, UsdPhysics, PhysxSchema
    if not prim or not prim.IsValid():
        return
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh) and collision_approximation is not None:
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        if not mesh_collision:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        approximation_attr = mesh_collision.GetApproximationAttr()
        if approximation_attr.IsValid():
            approximation_attr.Set(collision_approximation)
        else:
            mesh_collision.CreateApproximationAttr().Set(collision_approximation)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if not prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)
    # Kinematic flag is on UsdPhysics.RigidBodyAPI (version-safe)
    rb_api = UsdPhysics.RigidBodyAPI(prim)
    kin_attr = rb_api.GetKinematicEnabledAttr()
    if not kin_attr:
        kin_attr = rb_api.CreateKinematicEnabledAttr()
    kin_attr.Set(bool(kinematic))

    # Gravity disable is PhysX-specific; guard for API differences
    try:
        physx_body = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_body:
            physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        disable_g_attr = physx_body.GetDisableGravityAttr()
        if not disable_g_attr:
            disable_g_attr = physx_body.CreateDisableGravityAttr()
        disable_g_attr.Set(not enable_gravity)
    except Exception:
        pass


def _ensure_irregular_mesh(
    stage,
    mesh_path="/World/IrregularMesh",
    marker_path="/World/IrregularMarker",
    anchor_cube_path="/World/BigRayCube",
    seed=0,
    size=0.08,
):
    from pxr import UsdGeom, Gf, Usd
    prim = stage.GetPrimAtPath(mesh_path)
    if prim and prim.IsValid():
        _apply_rigidbody(
            prim,
            mass=1.0,
            enable_gravity=True,
            kinematic=False,
            collision_approximation="convexDecomposition",
        )
        return
    random.seed(seed)

    # Start from a cube and randomly perturb each corner
    half = size * 0.5
    base = [
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    ]
    points = []
    jitter = size * 0.35
    for x, y, z in base:
        points.append(
            Gf.Vec3f(
                x + random.uniform(-jitter, jitter),
                y + random.uniform(-jitter, jitter),
                z + random.uniform(-jitter, jitter),
            )
        )

    # 12 triangles (2 per face)
    face_counts = [3] * 12
    face_indices = [
        0, 1, 2, 0, 2, 3,  # -Z
        4, 6, 5, 4, 7, 6,  # +Z
        0, 4, 5, 0, 5, 1,  # -Y
        3, 2, 6, 3, 6, 7,  # +Y
        0, 3, 7, 0, 7, 4,  # -X
        1, 5, 6, 1, 6, 2,  # +X
    ]

    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.7, 0.9)])

    # Place it near the +X side of the cube (if available)
    cube_prim = stage.GetPrimAtPath(anchor_cube_path)
    if cube_prim and cube_prim.IsValid():
        xformable = UsdGeom.Xformable(cube_prim)
        xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        center = xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        # 放到立方体旁边 (+Y 方向)，避免与 Ray 网格重叠
        offset = Gf.Vec3d(0.0, size * 2.0, size * 0.6)
        mesh_pos = center + offset
        UsdGeom.XformCommonAPI(mesh).SetTranslate(mesh_pos)
    else:
        mesh_pos = Gf.Vec3d(0.0, 0.3, 0.1)
        UsdGeom.XformCommonAPI(mesh).SetTranslate(mesh_pos)
    _apply_rigidbody(
        mesh.GetPrim(),
        mass=1.0,
        enable_gravity=True,
        kinematic=False,
        collision_approximation="convexDecomposition",
    )

    # 同位置放一个球体用于观察/记录
    if not stage.GetPrimAtPath(marker_path).IsValid():
        marker = UsdGeom.Sphere.Define(stage, marker_path)
        marker.GetRadiusAttr().Set(max(0.01, size * 0.25))
        marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.3, 0.3)])
        UsdGeom.XformCommonAPI(marker).SetTranslate(mesh_pos)
        _apply_rigidbody(marker.GetPrim(), mass=0.5, enable_gravity=True, kinematic=False)

def _get_face_sign(face_selector: str) -> float:
    return 1.0 if str(face_selector).strip().lower() == "+x" else -1.0


def _ensure_ray_marker(
    stage,
    cube_path="/World/BigRayCube",
    marker_path=None,
    padding=0.002,
    face_selector="+x",
):
    from pxr import UsdGeom, Gf
    cube_prim = stage.GetPrimAtPath(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        return
    if marker_path is None:
        marker_path = f"{cube_path}/RayMarker"
    cube = UsdGeom.Cube(cube_prim)
    size = float(cube.GetSizeAttr().Get() or 0.2)
    half = size * 0.5
    face_sign = _get_face_sign(face_selector)
    x = face_sign * (half + padding)
    marker = UsdGeom.Sphere.Define(stage, marker_path)
    marker.GetRadiusAttr().Set(max(0.002, size * 0.03))
    marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.85, 0.2)])
    UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(x, 0.0, 0.0))


def _ensure_raycast_grid(
    stage,
    cube_path="/World/BigRayCube",
    grid_parent=None,
    grid_size=10,
    padding=0.002,
    face_selector="+x",
):
    from pxr import UsdGeom, Gf

    if grid_parent is None:
        grid_parent = f"{cube_path}/RayGrid"

    parent_prim = stage.GetPrimAtPath(grid_parent)
    if not parent_prim or not parent_prim.IsValid():
        parent_prim = UsdGeom.Xform.Define(stage, grid_parent).GetPrim()

    cube_prim = stage.GetPrimAtPath(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        return []

    cube = UsdGeom.Cube(cube_prim)
    size = float(cube.GetSizeAttr().Get() or 0.2)
    half = size * 0.5
    face_sign = _get_face_sign(face_selector)
    x = face_sign * (half + padding)

    span = size * 0.8
    start = -span * 0.5
    step = span / max(1, grid_size - 1)
    radius = size * 0.01

    origins_local = []
    for r in range(grid_size):
        for c in range(grid_size):
            y = start + step * r
            z = start + step * c
            local_pt = Gf.Vec3d(x, y, z)
            sphere_path = f"{grid_parent}/ray_{r}_{c}"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(radius)
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.95, 0.2)])
            UsdGeom.XformCommonAPI(sphere).SetTranslate(local_pt)
            origins_local.append((r, c, (float(local_pt[0]), float(local_pt[1]), float(local_pt[2]))))
    return origins_local


def _compute_raycast_frame(stage, cube_path, local_origins, face_selector):
    from pxr import UsdGeom, Gf, Usd

    cube_prim = stage.GetPrimAtPath(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        return [], None

    xform = UsdGeom.Xformable(cube_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    face_sign = _get_face_sign(face_selector)
    local_dir = Gf.Vec3d(face_sign, 0.0, 0.0)
    world_origin = xform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    world_dir_end = xform.Transform(local_dir)
    dir_vec = world_dir_end - world_origin
    dir_len = math.sqrt(float(dir_vec[0]) ** 2 + float(dir_vec[1]) ** 2 + float(dir_vec[2]) ** 2)
    if dir_len <= 1e-8:
        return [], None
    world_dir = (
        float(dir_vec[0]) / dir_len,
        float(dir_vec[1]) / dir_len,
        float(dir_vec[2]) / dir_len,
    )

    world_origins = []
    for r, c, local_origin in local_origins:
        local_pt = Gf.Vec3d(*local_origin)
        world_pt = xform.Transform(local_pt)
        world_origins.append((r, c, (float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))))
    return world_origins, world_dir


def _get_prim_world_transform(stage, prim_path):
    from pxr import UsdGeom, Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None

    world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_translation = world_xform.ExtractTranslation()
    world_rotation = world_xform.ExtractRotation().GetQuaternion()
    imag = world_rotation.GetImaginary()
    matrix_rows = []
    for row_idx in range(4):
        row = world_xform.GetRow(row_idx)
        matrix_rows.append([float(row[col_idx]) for col_idx in range(4)])

    return {
        "position": (
            float(world_translation[0]),
            float(world_translation[1]),
            float(world_translation[2]),
        ),
        "orientation_quat_wxyz": (
            float(world_rotation.GetReal()),
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
        ),
        "matrix4x4_row_major": matrix_rows,
    }


def _get_prim_world_matrix(stage, prim_path):
    from pxr import UsdGeom, Usd

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return None

    return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _as_float3(value):
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None


def _transform_point_to_local(stage, prim_path, world_point):
    from pxr import Gf

    if world_point is None:
        return None
    world_matrix = _get_prim_world_matrix(stage, prim_path)
    if world_matrix is None:
        return None
    local_point = world_matrix.GetInverse().Transform(Gf.Vec3d(*world_point))
    return (float(local_point[0]), float(local_point[1]), float(local_point[2]))


def _transform_point_to_world(stage, prim_path, local_point):
    from pxr import Gf

    if local_point is None:
        return None
    world_matrix = _get_prim_world_matrix(stage, prim_path)
    if world_matrix is None:
        return None
    world_point = world_matrix.Transform(Gf.Vec3d(*local_point))
    return (float(world_point[0]), float(world_point[1]), float(world_point[2]))


def _subtract_float3(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return (
        float(lhs[0]) - float(rhs[0]),
        float(lhs[1]) - float(rhs[1]),
        float(lhs[2]) - float(rhs[2]),
    )


def _dot_float3(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return (
        float(lhs[0]) * float(rhs[0])
        + float(lhs[1]) * float(rhs[1])
        + float(lhs[2]) * float(rhs[2])
    )


def _scale_float3(vec, scale):
    if vec is None or scale is None:
        return None
    return (
        float(vec[0]) * float(scale),
        float(vec[1]) * float(scale),
        float(vec[2]) * float(scale),
    )


def _add_float3(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return (
        float(lhs[0]) + float(rhs[0]),
        float(lhs[1]) + float(rhs[1]),
        float(lhs[2]) + float(rhs[2]),
    )


def _norm_float3(vec):
    if vec is None:
        return None
    return math.sqrt(
        float(vec[0]) * float(vec[0])
        + float(vec[1]) * float(vec[1])
        + float(vec[2]) * float(vec[2])
    )


def _safe_velocity(curr_vec, prev_vec, dt):
    if curr_vec is None or prev_vec is None or dt <= 1e-6:
        return (0.0, 0.0, 0.0)
    delta_vec = _subtract_float3(curr_vec, prev_vec)
    return (
        float(delta_vec[0]) / float(dt),
        float(delta_vec[1]) / float(dt),
        float(delta_vec[2]) / float(dt),
    )


def _project_out_normal_component(vec, normal):
    if vec is None or normal is None:
        return None
    normal_component = _scale_float3(normal, _dot_float3(vec, normal))
    return _subtract_float3(vec, normal_component)


def _get_face_normal_local(face_selector):
    face_sign = _get_face_sign(face_selector)
    return (float(face_sign), 0.0, 0.0)


def _is_target_body_contact(ray_row, max_distance, target_body_path):
    if not ray_row:
        return False
    if not bool(ray_row.get("hit")):
        return False
    if bool(ray_row.get("ignored_self_hit")):
        return False
    if str(ray_row.get("rigid_body")) != str(target_body_path):
        return False
    distance = ray_row.get("distance")
    if distance is None:
        return False
    return float(distance) < float(max_distance)


def _raycast_distances(
    physx_query,
    origins,
    direction,
    max_distance,
    ignore_rigid_body=None,
    collect_debug=False,
):
    distances = []
    ray_rows = []
    for r, c, origin in origins:
        hit = physx_query.raycast_closest(origin, direction, max_distance)
        hit_body = None
        hit_distance = None
        hit_position = None
        hit_normal = None
        hit_detected = bool(hit and hit.get("hit"))
        if hit_detected:
            hit_body = hit.get("rigidBody")
            hit_distance = float(hit.get("distance", 0.0))
            hit_position = _as_float3(hit.get("position"))
            hit_normal = _as_float3(hit.get("normal"))
        if hit and hit.get("hit"):
            if ignore_rigid_body and hit_body == ignore_rigid_body:
                distances.append((r, c, None))
            else:
                distances.append((r, c, hit_distance))
        else:
            distances.append((r, c, None))
        ray_rows.append(
            {
                "row": int(r),
                "col": int(c),
                "origin": tuple(float(v) for v in origin),
                "direction": tuple(float(v) for v in direction),
                "hit": hit_detected,
                "rigid_body": str(hit_body) if hit_body is not None else None,
                "distance": hit_distance,
                "position": hit_position,
                "normal": hit_normal,
                "ignored_self_hit": bool(ignore_rigid_body and hit_body == ignore_rigid_body),
                "collect_debug": bool(collect_debug),
            }
        )
    return distances, ray_rows


class PerTaxelContactTracker:
    def __init__(
        self,
        local_origins,
        target_body_path,
        sensor_parent_prim_path,
        sensor_face_selector,
        tangential_k,
        tangential_c,
    ):
        self.target_body_path = str(target_body_path)
        self.sensor_parent_prim_path = str(sensor_parent_prim_path)
        self.sensor_face_selector = str(sensor_face_selector)
        self.sensor_normal_local = _get_face_normal_local(sensor_face_selector)
        self.tangential_k = float(tangential_k)
        self.tangential_c = float(tangential_c)
        self.states = {}
        for r, c, local_origin in local_origins:
            key = (int(r), int(c))
            sensor_prim_path = f"{self.sensor_parent_prim_path}/RayGrid/ray_{int(r)}_{int(c)}"
            self.states[key] = {
                "row": int(r),
                "col": int(c),
                "sensor_prim_path": sensor_prim_path,
                "contact_active": False,
                "contact_started": False,
                "contact_ended": False,
                "last_event": "none",
                "target_body_path": self.target_body_path,
                "target_body_hit": False,
                "hit_rigid_body": None,
                "distance": None,
                "hit_point_world": None,
                "hit_normal_world": None,
                "sensor_origin_local": tuple(float(v) for v in local_origin),
                "pA_local": None,
                "pB_local": None,
                "p0_world": None,
                "pA_world": None,
                "pB_world": None,
                "d_world": None,
                "pB_in_A": None,
                "d_A": None,
                "d_t_A": None,
                "v_t_A": (0.0, 0.0, 0.0),
                "v_t_A_norm": 0.0,
                "xi_t_candidate": (0.0, 0.0, 0.0),
                "F_t_candidate": (0.0, 0.0, 0.0),
                "F_t_candidate_norm": 0.0,
                "sensor_normal_local": self.sensor_normal_local,
                "anchor_initialized": False,
                "contact_point_world": None,
                "xi_t": (0.0, 0.0, 0.0),
                "d_t_prev": (0.0, 0.0, 0.0),
                "last_timestamp": None,
                "contact_mode": None,
            }

    def update(self, ray_rows, timestamp, max_distance, stage):
        ray_row_map = {
            (int(row["row"]), int(row["col"])): row
            for row in ray_rows
        }
        now = float(timestamp)
        active_count = 0
        started_count = 0
        ended_count = 0
        min_active_distance = None
        states_payload = []

        for key in sorted(self.states):
            state = self.states[key]
            ray_row = ray_row_map.get(key)
            is_active = _is_target_body_contact(ray_row, max_distance, self.target_body_path)
            prev_active = bool(state["contact_active"])
            started = is_active and (not prev_active)
            ended = prev_active and (not is_active)

            if started:
                state["last_event"] = "start"
            elif ended:
                state["last_event"] = "end"
            elif is_active:
                state["last_event"] = "hold"
            else:
                state["last_event"] = "none"

            state["contact_active"] = is_active
            state["contact_started"] = started
            state["contact_ended"] = ended
            state["target_body_hit"] = bool(
                ray_row and bool(ray_row.get("hit")) and (str(ray_row.get("rigid_body")) == self.target_body_path)
            )
            state["hit_rigid_body"] = None if not ray_row else ray_row.get("rigid_body")
            state["distance"] = None if (not ray_row or ray_row.get("distance") is None) else float(ray_row["distance"])
            state["hit_point_world"] = None if not ray_row else ray_row.get("position")
            state["hit_normal_world"] = None if not ray_row else ray_row.get("normal")

            if is_active:
                state["contact_point_world"] = state["hit_point_world"]
                if (not state["anchor_initialized"]) and state["hit_point_world"] is not None:
                    p0_world = tuple(float(v) for v in state["hit_point_world"])
                    pA_local = _transform_point_to_local(stage, self.sensor_parent_prim_path, p0_world)
                    pB_local = _transform_point_to_local(stage, self.target_body_path, p0_world)
                    state["p0_world"] = p0_world
                    state["pA_local"] = pA_local
                    state["pB_local"] = pB_local
                    state["anchor_initialized"] = (pA_local is not None) and (pB_local is not None)
                if state["anchor_initialized"]:
                    state["pA_world"] = _transform_point_to_world(stage, self.sensor_parent_prim_path, state["pA_local"])
                    state["pB_world"] = _transform_point_to_world(stage, self.target_body_path, state["pB_local"])
                    state["d_world"] = _subtract_float3(state["pB_world"], state["pA_world"])
                    state["pB_in_A"] = _transform_point_to_local(stage, self.sensor_parent_prim_path, state["pB_world"])
                    state["d_A"] = _subtract_float3(state["pB_in_A"], state["pA_local"])
                    state["d_t_A"] = _project_out_normal_component(state["d_A"], state["sensor_normal_local"])
                    dt = 0.0 if state["last_timestamp"] is None else max(0.0, now - float(state["last_timestamp"]))
                    state["v_t_A"] = _safe_velocity(state["d_t_A"], state["d_t_prev"], dt)
                    state["v_t_A_norm"] = _norm_float3(state["v_t_A"]) or 0.0
                    state["xi_t_candidate"] = _add_float3(
                        state["xi_t"],
                        _scale_float3(state["v_t_A"], dt),
                    )
                    spring_term = _scale_float3(state["xi_t_candidate"], -self.tangential_k)
                    damping_term = _scale_float3(state["v_t_A"], -self.tangential_c)
                    state["F_t_candidate"] = _add_float3(spring_term, damping_term)
                    state["F_t_candidate_norm"] = _norm_float3(state["F_t_candidate"]) or 0.0
                    state["xi_t"] = state["xi_t_candidate"]
                    state["d_t_prev"] = state["d_t_A"] if state["d_t_A"] is not None else (0.0, 0.0, 0.0)
                else:
                    state["pA_world"] = None
                    state["pB_world"] = None
                    state["d_world"] = None
                    state["pB_in_A"] = None
                    state["d_A"] = None
                    state["d_t_A"] = None
                    state["v_t_A"] = (0.0, 0.0, 0.0)
                    state["v_t_A_norm"] = 0.0
                    state["xi_t_candidate"] = state["xi_t"]
                    state["F_t_candidate"] = (0.0, 0.0, 0.0)
                    state["F_t_candidate_norm"] = 0.0
                state["last_timestamp"] = now
                active_count += 1
                if state["distance"] is not None:
                    min_active_distance = (
                        state["distance"]
                        if min_active_distance is None
                        else min(min_active_distance, state["distance"])
                    )
            elif ended:
                state["p0_world"] = None
                state["contact_point_world"] = None
                state["pA_local"] = None
                state["pB_local"] = None
                state["pA_world"] = None
                state["pB_world"] = None
                state["d_world"] = None
                state["pB_in_A"] = None
                state["d_A"] = None
                state["d_t_A"] = None
                state["v_t_A"] = (0.0, 0.0, 0.0)
                state["v_t_A_norm"] = 0.0
                state["xi_t_candidate"] = (0.0, 0.0, 0.0)
                state["F_t_candidate"] = (0.0, 0.0, 0.0)
                state["F_t_candidate_norm"] = 0.0
                state["anchor_initialized"] = False
                state["xi_t"] = (0.0, 0.0, 0.0)
                state["d_t_prev"] = (0.0, 0.0, 0.0)
                # End the current contact episode completely so the next
                # re-contact starts with a fresh time base.
                state["last_timestamp"] = None
                state["contact_mode"] = None

            if started:
                started_count += 1
            if ended:
                ended_count += 1

            states_payload.append(dict(state))

        return {
            "target_body_path": self.target_body_path,
            "any_active": active_count > 0,
            "active_count": active_count,
            "started_count": started_count,
            "ended_count": ended_count,
            "min_active_distance": min_active_distance,
            "states": states_payload,
        }


def _distance_to_delta(distance, max_distance):
    # delta(压入量)公式:
    #   delta = max(0, max_distance - distance)
    # 说明:
    # - distance 为 ray 命中距离
    # - 无命中或超出 max_distance 视为无接触，delta=0
    if distance is None:
        return 0.0
    if distance > max_distance:
        return 0.0
    return max(0.0, max_distance - distance)


def _delta_to_force_legacy(delta_value, max_distance, force_max):
    # force 映射公式(线性归一化):
    #   force = (delta / max_distance) * force_max
    # 等价写法:
    #   force = max(0, (max_distance - distance) / max_distance) * force_max
    if delta_value is None:
        return 0.0
    return (delta_value / max(max_distance, 1e-6)) * force_max


def _delta_to_force_spring_damper(delta_value, delta_dot, spring_k, damping_c):
    # 法向力(弹簧+阻尼)公式:
    #   Fn = k * delta + c * delta_dot
    # 单边接触约束:
    #   无接触(delta<=0)时 Fn=0，且 Fn 不允许为负
    if delta_value is None or delta_value <= 0.0:
        return 0.0
    fn = float(spring_k) * float(delta_value) + float(damping_c) * float(delta_dot)
    return max(0.0, fn)


class RaySignalProcessor:
    """Lightweight stateful processor for distance/delta/delta_dot/force signals."""

    def __init__(
        self,
        max_distance: float,
        force_max: float,
        force_mapping: str,
        spring_k: float,
        damping_c: float,
    ):
        self.max_distance = float(max_distance)
        self.force_max = float(force_max)
        self.force_mapping = str(force_mapping)
        self.spring_k = float(spring_k)
        self.damping_c = float(damping_c)
        self.prev_timestamp = None
        self.prev_delta_map = {}

    @staticmethod
    def _delta_to_delta_dot(curr_delta: float, prev_delta: float, dt: float) -> float:
        # delta_dot(侵入速度)公式:
        #   delta_dot = (delta_t - delta_t-1) / dt
        # 若首帧或 dt 过小，则返回 0，避免除零与数值爆炸。
        if prev_delta is None or dt <= 1e-6:
            return 0.0
        return (curr_delta - prev_delta) / dt

    def update(self, distances, timestamp: float):
        now = float(timestamp)
        dt = 0.0 if self.prev_timestamp is None else max(0.0, now - self.prev_timestamp)
        records = []
        for r, c, d in distances:
            delta_value = _distance_to_delta(d, self.max_distance)
            prev_delta = self.prev_delta_map.get((int(r), int(c)))
            delta_dot = self._delta_to_delta_dot(delta_value, prev_delta, dt)
            if self.force_mapping == "legacy":
                force_value = _delta_to_force_legacy(delta_value, self.max_distance, self.force_max)
            else:
                force_value = _delta_to_force_spring_damper(
                    delta_value,
                    delta_dot,
                    self.spring_k,
                    self.damping_c,
                )
            records.append((r, c, d, delta_value, delta_dot, force_value))
        for r, c, _, delta_value, _, _ in records:
            self.prev_delta_map[(int(r), int(c))] = delta_value
        self.prev_timestamp = now
        return records, dt


def _format_stage_unit(meters_per_unit: float) -> str:
    if abs(meters_per_unit - 1.0) < 1e-9:
        return "meter (m)"
    if abs(meters_per_unit - 0.01) < 1e-9:
        return "centimeter (cm)"
    if abs(meters_per_unit - 0.001) < 1e-9:
        return "millimeter (mm)"
    return "custom unit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-open", action="store_true", help="保持窗口打开直到手动关闭")
    parser.add_argument("--close-after-frames", type=int, default=120, help="自动关闭前更新的帧数")
    parser.add_argument("--ray-grid-size", type=int, default=10, help="Ray 网格边长")
    parser.add_argument("--ray-max-distance", type=float, default=0.1, help="Ray 最大距离")
    parser.add_argument("--ray-padding", type=float, default=0.002, help="Ray 起点离表面的偏移(米)")
    parser.add_argument("--ray-update-interval", type=float, default=0.1, help="Ray 数据刷新间隔(秒)")
    parser.add_argument("--ray-log-dir", type=str, default="data/ray_logs", help="Ray 日志输出目录")
    parser.add_argument("--force-max", type=float, default=1.0, help="旧线性映射(legacy)的最大力")
    parser.add_argument(
        "--force-mapping",
        type=str,
        default="spring_damper",
        choices=["spring_damper", "legacy"],
        help="力映射模式: spring_damper(默认) 或 legacy",
    )
    parser.add_argument("--spring-k", type=float, default=10.0, help="弹簧刚度 k (N/m)")
    parser.add_argument("--damping-c", type=float, default=5.0, help="阻尼系数 c (N·s/m)")
    parser.add_argument("--tangential-k", type=float, default=10.0, help="切向弹簧刚度 k_t (N/m)")
    parser.add_argument("--tangential-c", type=float, default=5.0, help="切向阻尼系数 c_t (N·s/m)")
    parser.add_argument("--ray-direction", type=str, default="+x", help="Ray 方向: -x 或 +x")
    parser.add_argument("--irregular-seed", type=int, default=0, help="不规则网格随机种子")
    parser.add_argument("--irregular-size", type=float, default=0.08, help="不规则网格尺寸")
    parser.add_argument("--auto-view", action="store_true", help="自动启动外部可视化")
    parser.add_argument("--no-auto-view", action="store_true", help="不自动启动外部可视化")
    parser.add_argument("--view-vmin", type=float, default=0.0, help="可视化颜色最小值")
    parser.add_argument("--view-vmax", type=float, default=5.0, help="可视化颜色最大值")
    parser.add_argument("--view-cmap", type=str, default="inferno", help="可视化 colormap")
    parser.add_argument("--view-interval", type=float, default=0.1, help="可视化刷新间隔(秒)")
    parser.add_argument("--debug-ray", action="store_true", help="打印 Ray origin / hit rigidBody / distance 调试信息")
    parser.add_argument("--debug-ray-limit", type=int, default=5, help="每次调试输出最多打印多少条 Ray")
    parser.add_argument("--debug-ray-interval", type=float, default=1.0, help="Ray 调试输出节流间隔(秒)")
    parser.add_argument(
        "--target-contact-body-path",
        type=str,
        default="/World/Cylinder_Test",
        help="持续跟踪的目标刚体 prim path",
    )
    parser.add_argument(
        "--usd-path",
        type=str,
        default=None,
        help="USD 场景路径；支持绝对路径，或相对当前目录/仓库根目录的路径",
    )
    args = parser.parse_args()

    # Launch Isaac Sim and open the USD stage
    simulation_app = SimulationApp({"headless": False})

    # Import after SimulationApp is initialized
    import omni.usd

    usd_path = _get_usd_path(args.usd_path)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    # Open the USD stage via omni.usd context
    ctx = omni.usd.get_context()
    opened = ctx.open_stage(usd_path)
    if not opened:
        raise RuntimeError(f"无法打开 USD: {usd_path}")

    # Wait until stage finishes loading (API differences across Isaac Sim versions)
    if hasattr(ctx, "is_loading"):
        while ctx.is_loading():
            simulation_app.update()
    elif hasattr(ctx, "get_stage_state"):
        import omni.usd as _omni_usd
        for _ in range(600):
            state = ctx.get_stage_state()
            if "OPEN" in str(state):
                break
            simulation_app.update()
    else:
        # Fallback: advance a few frames and ensure stage exists
        for _ in range(120):
            simulation_app.update()

    # Ensure stage is available
    for _ in range(120):
        if ctx.get_stage() is not None:
            break
        simulation_app.update()

    stage = ctx.get_stage()
    cube_path = "/World/BigRayCube"
    ray_local_origins = []
    if stage is not None:
        try:
            from pxr import UsdGeom
            meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
            unit_label = _format_stage_unit(meters_per_unit)
            # 启动时输出场景尺度：1 个 USD 长度单位等于多少米（用于判断 Raycast 距离单位）
            print(
                f"[INFO] Stage unit scale: 1 unit = {meters_per_unit} m ({unit_label})"
            )
        except Exception as exc:
            print(f"[WARN] Failed to read stage unit scale: {exc}")

        cube_prim = _ensure_big_cube(stage, cube_path=cube_path)
        _apply_rigidbody(cube_prim, mass=2.0, enable_gravity=True, kinematic=False)
        _ensure_ray_marker(
            stage,
            cube_path=cube_path,
            padding=float(args.ray_padding),
            face_selector=args.ray_direction,
        )
        ray_local_origins = _ensure_raycast_grid(
            stage,
            cube_path=cube_path,
            grid_size=args.ray_grid_size,
            padding=float(args.ray_padding),
            face_selector=args.ray_direction,
        ) or []

    run_id = time.strftime("%Y%m%d_%H%M%S")
    ray_log_dir = _resolve_repo_path(args.ray_log_dir)
    os.makedirs(ray_log_dir, exist_ok=True)
    ray_log_path = os.path.join(ray_log_dir, f"ray_distances_{run_id}.json")
    ray_log_csv = os.path.join(ray_log_dir, f"ray_distances_{run_id}.csv")
    force_log_csv = os.path.join(ray_log_dir, f"ray_forces_{run_id}.csv")
    delta_log_csv = os.path.join(ray_log_dir, f"ray_deltas_{run_id}.csv")
    delta_dot_log_csv = os.path.join(ray_log_dir, f"ray_delta_dots_{run_id}.csv")
    prim_pose_log_csv = os.path.join(ray_log_dir, f"prim_world_positions_{run_id}.csv")
    prim_transform_log_csv = os.path.join(ray_log_dir, f"prim_world_transforms_{run_id}.csv")
    contact_state_log_csv = os.path.join(ray_log_dir, f"contact_states_{run_id}.csv")
    taxel_contact_log_csv = os.path.join(ray_log_dir, f"taxel_contacts_{run_id}.csv")
    target_contact_body_path = str(args.target_contact_body_path)
    tracked_prim_paths = [target_contact_body_path, "/World/BigRayCube"]
    taxel_contact_tracker = PerTaxelContactTracker(
        local_origins=ray_local_origins,
        target_body_path=target_contact_body_path,
        sensor_parent_prim_path=cube_path,
        sensor_face_selector=args.ray_direction,
        tangential_k=args.tangential_k,
        tangential_c=args.tangential_c,
    )

    # Auto-start external visualization (default on unless --no-auto-view)
    auto_view = args.auto_view or not args.no_auto_view
    view_proc = None
    if auto_view:
        view_script = os.path.join(os.path.dirname(__file__), "ray_live_view.py")
        cmd = [
            sys.executable,
            view_script,
            "--json",
            ray_log_path,
            "--vmin",
            str(args.view_vmin),
            "--vmax",
            str(args.view_vmax),
            "--cmap",
            args.view_cmap,
            "--interval",
            str(args.view_interval),
            "--parent-pid",
            str(os.getpid()),
        ]
        try:
            view_proc = subprocess.Popen(cmd)
        except Exception:
            pass

    def _cleanup_view():
        if view_proc and view_proc.poll() is None:
            try:
                view_proc.terminate()
                view_proc.wait(timeout=2.0)
            except Exception:
                try:
                    view_proc.kill()
                except Exception:
                    pass

    atexit.register(_cleanup_view)
    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    import omni.physx
    physx_query = omni.physx.get_physx_scene_query_interface()
    last_ray_update = 0.0
    last_debug_print = 0.0
    csv_header_written = False
    force_header_written = False
    delta_header_written = False
    delta_dot_header_written = False
    prim_pose_header_written = False
    prim_transform_header_written = False
    contact_state_header_written = False
    taxel_contact_header_written = False
    signal_processor = RaySignalProcessor(
        max_distance=float(args.ray_max_distance),
        force_max=float(args.force_max),
        force_mapping=str(args.force_mapping),
        spring_k=float(args.spring_k),
        damping_c=float(args.damping_c),
    )

    def _update_and_log(now: float) -> None:
        nonlocal last_ray_update
        nonlocal last_debug_print
        nonlocal csv_header_written
        nonlocal force_header_written
        nonlocal delta_header_written
        nonlocal delta_dot_header_written
        nonlocal prim_pose_header_written
        nonlocal prim_transform_header_written
        nonlocal contact_state_header_written
        nonlocal taxel_contact_header_written

        world_ray_origins, world_ray_dir = _compute_raycast_frame(
            stage,
            cube_path,
            ray_local_origins,
            args.ray_direction,
        )
        if not world_ray_origins or world_ray_dir is None:
            return
        distances, debug_rows = _raycast_distances(
            physx_query,
            world_ray_origins,
            world_ray_dir,
            args.ray_max_distance,
            ignore_rigid_body=cube_path,
            collect_debug=bool(args.debug_ray),
        )
        records, dt = signal_processor.update(distances, now)
        taxel_contact_snapshot = taxel_contact_tracker.update(
            debug_rows,
            now,
            args.ray_max_distance,
            stage,
        )
        contact_active = bool(taxel_contact_snapshot["any_active"])
        min_contact_distance = taxel_contact_snapshot["min_active_distance"]
        contact_ray_count = int(taxel_contact_snapshot["active_count"])
        if args.debug_ray:
            record_map = {
                (int(r), int(c)): {
                    "distance": d,
                    "delta": delta,
                    "delta_dot": delta_dot,
                    "force": force,
                }
                for r, c, d, delta, delta_dot, force in records
            }
        if args.debug_ray and (now - last_debug_print >= max(0.0, float(args.debug_ray_interval))):
            non_none_distances = [d for _, _, d in distances if d is not None]
            non_zero_deltas = [
                row["delta"]
                for row in record_map.values()
                if row["delta"] is not None and float(row["delta"]) > 0.0
            ]
            print(
                "[DEBUG] Ray summary: "
                f"total={len(distances)} "
                f"hits={sum(1 for row in debug_rows if row['hit'])} "
                f"valid_hits={len(non_none_distances)} "
                f"contact_active={contact_active} "
                f"contact_rays={contact_ray_count} "
                f"contact_started={taxel_contact_snapshot['started_count']} "
                f"contact_ended={taxel_contact_snapshot['ended_count']} "
                f"ignored_self_hits={sum(1 for row in debug_rows if row['ignored_self_hit'])} "
                f"dir={tuple(round(v, 6) for v in world_ray_dir)}"
            )
            if non_none_distances:
                print(
                    "[DEBUG] Ray distances: "
                    f"min={min(non_none_distances):.6f} "
                    f"max={max(non_none_distances):.6f}"
                )
            else:
                print("[DEBUG] Ray distances: no valid external hits")
            if non_zero_deltas:
                print(
                    "[DEBUG] Ray deltas: "
                    f"min={min(non_zero_deltas):.6f} "
                    f"max={max(non_zero_deltas):.6f}"
                )
            else:
                print("[DEBUG] Ray deltas: all zero")
            for row in debug_rows[: max(0, int(args.debug_ray_limit))]:
                record = record_map.get((row["row"], row["col"]), {})
                print(
                    "[DEBUG] Ray sample "
                    f"r={row['row']} c={row['col']} "
                    f"origin={tuple(round(v, 6) for v in row['origin'])} "
                    f"dir={tuple(round(v, 6) for v in row['direction'])} "
                    f"hit={row['hit']} "
                    f"rigidBody={row['rigid_body']} "
                    f"distance={None if row['distance'] is None else round(row['distance'], 6)} "
                    f"delta={None if 'delta' not in record else round(float(record['delta']), 6)} "
                    f"delta_dot={None if 'delta_dot' not in record else round(float(record['delta_dot']), 6)} "
                    f"force={None if 'force' not in record else round(float(record['force']), 6)} "
                    f"ignored_self_hit={row['ignored_self_hit']}"
                )
            last_debug_print = now
        prim_world_transforms = {
            prim_path: _get_prim_world_transform(stage, prim_path)
            for prim_path in tracked_prim_paths
        }
        payload = {
            "timestamp": now,
            "delta_t": dt,
            "contact_active": contact_active,
            "contact_min_distance": min_contact_distance,
            "contact_ray_count": contact_ray_count,
            "contact_target_body_path": target_contact_body_path,
            "taxel_contact_summary": {
                "any_active": taxel_contact_snapshot["any_active"],
                "active_count": taxel_contact_snapshot["active_count"],
                "started_count": taxel_contact_snapshot["started_count"],
                "ended_count": taxel_contact_snapshot["ended_count"],
                "min_active_distance": taxel_contact_snapshot["min_active_distance"],
            },
            "grid_size": args.ray_grid_size,
            "max_distance": args.ray_max_distance,
            "direction": args.ray_direction,
            "direction_world": world_ray_dir,
            "force_mapping": args.force_mapping,
            "spring_k": args.spring_k,
            "damping_c": args.damping_c,
            "tangential_k": args.tangential_k,
            "tangential_c": args.tangential_c,
            "distances": distances,
            "deltas": [(r, c, delta) for r, c, _, delta, _, _ in records],
            "delta_dots": [(r, c, delta_dot) for r, c, _, _, delta_dot, _ in records],
            "forces": [(r, c, f) for r, c, _, _, _, f in records],
            "prim_world_positions": {
                prim_path: (None if prim_world_transforms[prim_path] is None else prim_world_transforms[prim_path]["position"])
                for prim_path in tracked_prim_paths
            },
            "prim_world_transforms": prim_world_transforms,
            "taxel_d_in_A": [
                (state["row"], state["col"], state["d_A"])
                for state in taxel_contact_snapshot["states"]
            ],
            "taxel_d_t_in_A": [
                (state["row"], state["col"], state["d_t_A"])
                for state in taxel_contact_snapshot["states"]
            ],
            "taxel_v_t_in_A": [
                (state["row"], state["col"], state["v_t_A"])
                for state in taxel_contact_snapshot["states"]
            ],
            "taxel_F_t_candidate": [
                (state["row"], state["col"], state["F_t_candidate"])
                for state in taxel_contact_snapshot["states"]
            ],
            "taxel_contact_states": taxel_contact_snapshot["states"],
        }
        with open(ray_log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with open(ray_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_header_written:
                writer.writerow(["timestamp", "row", "col", "distance"])
                csv_header_written = True
            for r, c, d, _, _, _ in records:
                writer.writerow([now, r, c, "" if d is None else d])
        with open(delta_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not delta_header_written:
                writer.writerow(["timestamp", "row", "col", "delta"])
                delta_header_written = True
            for r, c, _, delta, _, _ in records:
                writer.writerow([now, r, c, delta])
        with open(delta_dot_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not delta_dot_header_written:
                writer.writerow(["timestamp", "row", "col", "delta_dot"])
                delta_dot_header_written = True
            for r, c, _, _, delta_dot, _ in records:
                writer.writerow([now, r, c, delta_dot])
        with open(force_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not force_header_written:
                writer.writerow(["timestamp", "row", "col", "force"])
                force_header_written = True
            for r, c, _, _, _, force in records:
                writer.writerow([now, r, c, force])
        with open(prim_pose_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not prim_pose_header_written:
                writer.writerow(["timestamp", "prim_path", "x", "y", "z"])
                prim_pose_header_written = True
            for prim_path, world_pos in payload["prim_world_positions"].items():
                if world_pos is None:
                    writer.writerow([now, prim_path, "", "", ""])
                else:
                    writer.writerow([now, prim_path, world_pos[0], world_pos[1], world_pos[2]])
        with open(prim_transform_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not prim_transform_header_written:
                writer.writerow(
                    [
                        "timestamp",
                        "prim_path",
                        "px",
                        "py",
                        "pz",
                        "qw",
                        "qx",
                        "qy",
                        "qz",
                        "m00",
                        "m01",
                        "m02",
                        "m03",
                        "m10",
                        "m11",
                        "m12",
                        "m13",
                        "m20",
                        "m21",
                        "m22",
                        "m23",
                        "m30",
                        "m31",
                        "m32",
                        "m33",
                    ]
                )
                prim_transform_header_written = True
            for prim_path, world_transform in payload["prim_world_transforms"].items():
                if world_transform is None:
                    writer.writerow([now, prim_path] + [""] * 23)
                    continue
                position = world_transform["position"]
                quat = world_transform["orientation_quat_wxyz"]
                matrix_flat = [
                    value
                    for row in world_transform["matrix4x4_row_major"]
                    for value in row
                ]
                writer.writerow([now, prim_path, position[0], position[1], position[2], quat[0], quat[1], quat[2], quat[3]] + matrix_flat)
        with open(contact_state_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not contact_state_header_written:
                writer.writerow(
                    [
                        "timestamp",
                        "contact_active",
                        "contact_min_distance",
                        "contact_ray_count",
                        "contact_started_count",
                        "contact_ended_count",
                        "target_body_path",
                        "ray_max_distance",
                    ]
                )
                contact_state_header_written = True
            writer.writerow(
                [
                    now,
                    int(bool(contact_active)),
                    "" if min_contact_distance is None else min_contact_distance,
                    contact_ray_count,
                    taxel_contact_snapshot["started_count"],
                    taxel_contact_snapshot["ended_count"],
                    target_contact_body_path,
                    args.ray_max_distance,
                ]
            )
        with open(taxel_contact_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not taxel_contact_header_written:
                writer.writerow(
                    [
                        "timestamp",
                        "row",
                        "col",
                        "sensor_prim_path",
                        "contact_active",
                        "contact_started",
                        "contact_ended",
                        "last_event",
                        "target_body_path",
                        "target_body_hit",
                        "hit_rigid_body",
                        "distance",
                        "hit_px",
                        "hit_py",
                        "hit_pz",
                        "normal_x",
                        "normal_y",
                        "normal_z",
                        "sensor_origin_local_x",
                        "sensor_origin_local_y",
                        "sensor_origin_local_z",
                        "p0_world_x",
                        "p0_world_y",
                        "p0_world_z",
                        "pA_local_x",
                        "pA_local_y",
                        "pA_local_z",
                        "pB_local_x",
                        "pB_local_y",
                        "pB_local_z",
                        "pA_world_x",
                        "pA_world_y",
                        "pA_world_z",
                        "pB_world_x",
                        "pB_world_y",
                        "pB_world_z",
                        "d_world_x",
                        "d_world_y",
                        "d_world_z",
                        "sensor_normal_local_x",
                        "sensor_normal_local_y",
                        "sensor_normal_local_z",
                        "pB_in_A_x",
                        "pB_in_A_y",
                        "pB_in_A_z",
                        "d_A_x",
                        "d_A_y",
                        "d_A_z",
                        "d_t_A_x",
                        "d_t_A_y",
                        "d_t_A_z",
                        "v_t_A_x",
                        "v_t_A_y",
                        "v_t_A_z",
                        "v_t_A_norm",
                        "xi_t_x",
                        "xi_t_y",
                        "xi_t_z",
                        "xi_t_candidate_x",
                        "xi_t_candidate_y",
                        "xi_t_candidate_z",
                        "F_t_candidate_x",
                        "F_t_candidate_y",
                        "F_t_candidate_z",
                        "F_t_candidate_norm",
                        "anchor_initialized",
                        "last_timestamp",
                        "contact_mode",
                    ]
                )
                taxel_contact_header_written = True
            for state in taxel_contact_snapshot["states"]:
                hit_point_world = state["hit_point_world"]
                hit_normal_world = state["hit_normal_world"]
                sensor_origin_local = state["sensor_origin_local"]
                p0_world = state["p0_world"]
                pA_local = state["pA_local"]
                pB_local = state["pB_local"]
                pA_world = state["pA_world"]
                pB_world = state["pB_world"]
                d_world = state["d_world"]
                sensor_normal_local = state["sensor_normal_local"]
                pB_in_A = state["pB_in_A"]
                d_A = state["d_A"]
                d_t_A = state["d_t_A"]
                v_t_A = state["v_t_A"]
                xi_t = state["xi_t"]
                xi_t_candidate = state["xi_t_candidate"]
                F_t_candidate = state["F_t_candidate"]
                writer.writerow(
                    [
                        now,
                        state["row"],
                        state["col"],
                        state["sensor_prim_path"],
                        int(bool(state["contact_active"])),
                        int(bool(state["contact_started"])),
                        int(bool(state["contact_ended"])),
                        state["last_event"],
                        state["target_body_path"],
                        int(bool(state["target_body_hit"])),
                        state["hit_rigid_body"],
                        "" if state["distance"] is None else state["distance"],
                        "" if hit_point_world is None else hit_point_world[0],
                        "" if hit_point_world is None else hit_point_world[1],
                        "" if hit_point_world is None else hit_point_world[2],
                        "" if hit_normal_world is None else hit_normal_world[0],
                        "" if hit_normal_world is None else hit_normal_world[1],
                        "" if hit_normal_world is None else hit_normal_world[2],
                        sensor_origin_local[0],
                        sensor_origin_local[1],
                        sensor_origin_local[2],
                        "" if p0_world is None else p0_world[0],
                        "" if p0_world is None else p0_world[1],
                        "" if p0_world is None else p0_world[2],
                        "" if pA_local is None else pA_local[0],
                        "" if pA_local is None else pA_local[1],
                        "" if pA_local is None else pA_local[2],
                        "" if pB_local is None else pB_local[0],
                        "" if pB_local is None else pB_local[1],
                        "" if pB_local is None else pB_local[2],
                        "" if pA_world is None else pA_world[0],
                        "" if pA_world is None else pA_world[1],
                        "" if pA_world is None else pA_world[2],
                        "" if pB_world is None else pB_world[0],
                        "" if pB_world is None else pB_world[1],
                        "" if pB_world is None else pB_world[2],
                        "" if d_world is None else d_world[0],
                        "" if d_world is None else d_world[1],
                        "" if d_world is None else d_world[2],
                        sensor_normal_local[0],
                        sensor_normal_local[1],
                        sensor_normal_local[2],
                        "" if pB_in_A is None else pB_in_A[0],
                        "" if pB_in_A is None else pB_in_A[1],
                        "" if pB_in_A is None else pB_in_A[2],
                        "" if d_A is None else d_A[0],
                        "" if d_A is None else d_A[1],
                        "" if d_A is None else d_A[2],
                        "" if d_t_A is None else d_t_A[0],
                        "" if d_t_A is None else d_t_A[1],
                        "" if d_t_A is None else d_t_A[2],
                        v_t_A[0],
                        v_t_A[1],
                        v_t_A[2],
                        state["v_t_A_norm"],
                        xi_t[0],
                        xi_t[1],
                        xi_t[2],
                        xi_t_candidate[0],
                        xi_t_candidate[1],
                        xi_t_candidate[2],
                        F_t_candidate[0],
                        F_t_candidate[1],
                        F_t_candidate[2],
                        state["F_t_candidate_norm"],
                        int(bool(state["anchor_initialized"])),
                        "" if state["last_timestamp"] is None else state["last_timestamp"],
                        state["contact_mode"],
                    ]
                )
        last_ray_update = now

    try:
        if args.keep_open:
            while (not stop_requested) and simulation_app.is_running():
                simulation_app.update()
                now = time.time()
                if ray_local_origins and (now - last_ray_update >= args.ray_update_interval):
                    _update_and_log(now)
        else:
            # Keep the app alive for a short time to ensure the stage is visible
            for _ in range(max(1, int(args.close_after_frames))):
                if stop_requested:
                    break
                simulation_app.update()
                now = time.time()
                if ray_local_origins and (now - last_ray_update >= args.ray_update_interval):
                    _update_and_log(now)
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        _cleanup_view()


if __name__ == "__main__":
    main()
