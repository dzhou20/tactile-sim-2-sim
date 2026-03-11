"""
Unit test for basic tactile (contact sensor) functionality.

Adds a helper that returns structured contact data (num_contacts, max_force, raw
frame) so callers can consume the sensor output programmatically instead of
only printing to stdout.
"""
import os
import time
from collections import defaultdict
import math
from dataclasses import dataclass
import traceback
import numpy as np

from isaacsim.simulation_app import SimulationApp

# 使用可视化模式运行，方便观察触觉网格
simulation_app = SimulationApp({"headless": False})

# Isaac Sim 相关模块（需在 SimulationApp 初始化后导入）
import omni.kit.commands
import omni.timeline
import omni.ui
import matplotlib
import omni.physx
from omni.ui_scene import scene as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
import omni.usd
from isaacsim.sensors.physics import _sensor

USD_PATH = "/home/dzhou20/epfl/sp_tactile_CREATE/create_asset/ur3_umi_unittest.usd"
UNIT_TEST_CUBES = [("/ur3/Cube_UnitTest1", (0.85, 0.25, 0.3)), ("/ur3/Cube_UnitTest2", (0.2, 0.65, 0.35))]
stage = None
# -----------------------------------------------------------------------------
# Global sensor settings (used by both cube and finger sensors)
# -----------------------------------------------------------------------------
CONTACT_SENSOR_RADIUS = 0.006          # Sensor sphere radius (meters)
CONTACT_SENSOR_QUERY_RADIUS = 0.05     # Query radius when reading sensor frames
CONTACT_SENSOR_MIN_THRESHOLD = -1.0        # Minimum force (N) to count as contact
CONTACT_SENSOR_NAME = "CubeContactSensor"
CONTACT_SENSOR_NAME_NEG_Y = "CubeContactSensorNegY"

# -----------------------------------------------------------------------------
# Cube contact grid settings (per selected face)
# -----------------------------------------------------------------------------
GRID_SIZE = 3                          # Grid size per face (GRID_SIZE x GRID_SIZE)
GRID_PADDING = 0.002                   # Offset along face normal (meters)
GRID_INSET = 0.002                     # Inset from face boundary (meters)
CONTACT_FACES_CUBE1 = ["pos_y"]        # Faces to instrument on Cube1
CONTACT_FACES_CUBE2 = ["neg_y"]        # Faces to instrument on Cube2

# -----------------------------------------------------------------------------
# Finger1 contact grid settings (custom YxZ grid on finger1_1 local frame)
# -----------------------------------------------------------------------------
ENABLE_FINGER1_SENSORS = True
FINGER1_PARENT = "/ur3/umi/finger1_1"
FINGER1_GRID_NY = 10                   # Count along +Y
FINGER1_GRID_NZ = 4                    # Count along +Z
FINGER1_GRID_STEP_Y = 0.004            # Spacing along +Y (meters)
FINGER1_GRID_STEP_Z = 0.002            # Spacing along +Z (meters)
FINGER1_X_OFFSET = 0.0                 # Fixed X offset (meters)
FINGER1_Y_OFFSET = 0.0                 # Base Y offset (meters)
FINGER1_Z_OFFSET = 0.0                 # Base Z offset (meters)
FINGER1_SENSOR_PREFIX = "Finger1Contact"

# -----------------------------------------------------------------------------
# Cube1 pos_z blocks (red cubes with sensors on top)
# -----------------------------------------------------------------------------
ENABLE_CUBE1_POSZ_BLOCKS = True
CUBE1_POSZ_PARENT = "/ur3/Cube_UnitTest1"
CUBE1_POSZ_BLOCK_GRID = (10, 10)       # (rows, cols) on pos_z face
CUBE1_POSZ_BLOCK_SIZE = 0.004          # Cube edge length (meters)
CUBE1_POSZ_BLOCK_PADDING = 0.001       # Extra offset from cube face (meters)
CUBE1_POSZ_BLOCK_COLOR = (0.4, 0.05, 0.05)  # Dark red (darker than Cube1)
CUBE1_POSZ_BLOCK_PREFIX = "Cube1PosZBlock"
CUBE1_POSZ_SENSOR_PREFIX = "Cube1PosZContact"
OLD_CUBE1_POSZ_CYL_PREFIX = "Cube1PosZCyl"

# -----------------------------------------------------------------------------
# UI and visualization settings
# -----------------------------------------------------------------------------
UI_UPDATE_INTERVAL = 0.1               # UI refresh interval (seconds)
VISUALIZE_SENSORS = False              # Enable 3D sensor spheres (global)
VISUALIZE_BLOCK_SPHERES = False        # Disable spheres for cube1 pos_z blocks
VIZ_RADIUS_SCALE = 1.2                 # Scale for visualization spheres
VIZ_COLOR_OPEN = (0.9, 0.2, 0.2)       # Red: no contact
VIZ_COLOR_CONTACT = (0.2, 0.9, 0.2)    # Green: in contact
RAYCAST_UI_FONT_SIZE = 12
RAYCAST_UI_TEXT_COLOR = (0.95, 0.95, 0.95, 1.0)
RAYCAST_UI_VALUE_FORMAT = "{:.3f}"
RAYCAST_UI_EMPTY_TEXT = "-"
VISUALIZE_FINGER1_RAYCAST_ORIGINS = True
FINGER1_RAYCAST_ORIGIN_SPHERE_RADIUS = 0.0015
FINGER1_RAYCAST_ORIGIN_COLOR = (0.2, 0.6, 0.95)
FINGER1_RAYCAST_ORIGIN_PARENT = "/ur3/umi/finger1_1/raycast_origins"

# -----------------------------------------------------------------------------
# Plot settings (force-time curves for pos_y sensors)
# -----------------------------------------------------------------------------
PLOT_POS_Y_FORCE = True
PLOT_SAMPLE_INTERVAL = 0.05            # Seconds between samples
PLOT_MIN_FORCE = -1e-6                 # Only plot points with force > this
PLOT_MAX_DURATION = 10.0               # Max duration to record (seconds)
PLOT_OUTPUT_PATH = (
    "/home/dzhou20/epfl/sp_tactile_CREATE/TacEx/create_hardware_align/pos_y_force_time.png"
)
PLOT_CONTACT_OUTPUT_PATH = (
    "/home/dzhou20/epfl/sp_tactile_CREATE/TacEx/create_hardware_align/pos_y_contact_time.png"
)
PLOT_COMBINED_OUTPUT_PATH = (
    "/home/dzhou20/epfl/sp_tactile_CREATE/TacEx/create_hardware_align/pos_y_contact_force.png"
)

# -----------------------------------------------------------------------------
# Simulation control
# -----------------------------------------------------------------------------
AUTO_PLAY = True                      # Start playing immediately on launch
ENABLE_UMI_GRIPPER_OPEN = True         # Move finger1_1 and finger2_1 after play starts
UMI_GRIPPER_OPEN_DURATION = 2.0        # Seconds to complete the motion
UMI_FINGER1_X_DELTA = 0.04             # finger1_1 local +X offset
UMI_FINGER2_X_DELTA = -0.04            # finger2_1 local +X offset

# -----------------------------------------------------------------------------
# Raycast settings (Cube_UnitTest2_Smaller -> front distance)
# -----------------------------------------------------------------------------
ENABLE_RAYCAST_GRID = True
RAYCAST_PARENT = "/ur3/Cube_UnitTest2_Smaller"
RAYCAST_FACE_LABEL = "pos_y"
RAYCAST_GRID_SIZE = 7
RAYCAST_PADDING = 0.001
RAYCAST_MAX_DISTANCE = 0.1
RAYCAST_UI_UPDATE_INTERVAL = 0.1
RAYCAST_COLOR_NO_HIT = (0.2, 0.2, 0.2)

# -----------------------------------------------------------------------------
# Raycast settings (finger1_1 -> clamp direction)
# -----------------------------------------------------------------------------
ENABLE_FINGER1_RAYCAST_GRID = True
FINGER1_RAYCAST_PARENT = "/ur3/umi/finger1_1"
FINGER1_RAYCAST_FACE_LABEL = "neg_x"
FINGER1_RAYCAST_GRID = (7, 5)         # (rows, cols)
FINGER1_RAYCAST_PADDING = 0.001
FINGER1_RAYCAST_MAX_DISTANCE = 0.02
FINGER1_RAYCAST_UI_UPDATE_INTERVAL = 0.1

# -----------------------------------------------------------------------------
# Real-time force plot settings (omni.ui)
# -----------------------------------------------------------------------------
ENABLE_REALTIME_FORCE_PLOT = True
FORCE_PLOT_HISTORY = 120              # Number of samples per curve
FORCE_PLOT_AUTO_SCALE = True
FORCE_PLOT_Y_MAX = 5.0                # Used when auto scale is off
FORCE_PLOT_UPDATE_INTERVAL = 0.1
contact_sensor = None
contact_sensor_cube2_neg_y = None
cube1_sensors = {}
cube2_sensors = {}
physics_scene = None
contact_sensor_iface = _sensor.acquire_contact_sensor_interface()
physx_scene_query = omni.physx.get_physx_scene_query_interface()
gravity_dir = Gf.Vec3d(0.0, 0.0, -1.0)
gravity_mag = 981.0
initial_time = None
sensor_viz_map = {}
FACE_DIRECTIONS = {
    "pos_x": (1, 0, 0),
    "neg_x": (-1, 0, 0),
    "pos_y": (0, 1, 0),
    "neg_y": (0, -1, 0),
    "pos_z": (0, 0, 1),
    "neg_z": (0, 0, -1),
}


@dataclass
class CsSensorReading:
    is_valid: bool
    time: float
    value: float
    in_contact: bool


def open_stage():
    """打开指定 USD，返回 stage。"""
    ctx = omni.usd.get_context()
    opened = ctx.open_stage(USD_PATH)
    if not opened:
        raise RuntimeError(f"无法打开 USD: {USD_PATH}")
    print(f"[INFO] Loaded USD file: {USD_PATH}")
    return ctx.get_stage()


def get_prim(path: str):
    if stage is None:
        return None
    try:
        sdf_path = path if isinstance(path, Sdf.Path) else Sdf.Path(path)
    except Exception:
        return None
    return stage.GetPrimAtPath(sdf_path)


def set_prim_color(prim_path: str, color):
    """为指定 prim 设置 displayColor。"""
    prim = get_prim(prim_path)
    if not prim or not prim.IsValid():
        print(f"[WARN] 找不到 prim: {prim_path}")
        return
    try:
        geom = UsdGeom.Gprim(prim)
        geom.CreateDisplayColorAttr().Set([color])
        print(f"[INFO] 已更新 {prim_path} 颜色为 {color}")
    except Exception as exc:
        print(f"[WARN] 无法设置 {prim_path} 颜色: {exc}")


def print_prim_pose_and_extent(prim_paths):
    """打印 prim 的局部/世界位姿与本地 AABB。"""
    for path in prim_paths:
        prim = get_prim(path)
        if not prim or not prim.IsValid():
            print(f"[WARN] prim not found: {path}")
            continue

        xform = UsdGeom.Xformable(prim)
        local_transform = xform.GetLocalTransformation()
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        local_t = local_transform.ExtractTranslation()
        world_t = world_transform.ExtractTranslation()
        local_r = local_transform.ExtractRotation()
        world_r = world_transform.ExtractRotation()

        extent = None
        if prim.IsA(UsdGeom.Boundable):
            extent = UsdGeom.Boundable(prim).ComputeExtent(Usd.TimeCode.Default())

        print(f"\n[INFO] {path}")
        print(f"  local translation: {local_t}")
        print(f"  world translation: {world_t}")
        print(f"  local rotation: {local_r.GetQuaternion()}")
        print(f"  world rotation: {world_r.GetQuaternion()}")
        print(f"  extent (local AABB): {extent}")


def print_boundable_extents_under(root_path: str):
    """递归打印 root_path 下所有可计算 extent 的几何 prim。"""
    root = get_prim(root_path)
    if not root or not root.IsValid():
        print(f"[WARN] prim not found: {root_path}")
        return
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Boundable):
            extent = UsdGeom.Boundable(prim).ComputeExtent(Usd.TimeCode.Default())
            print(f"[BOUNDABLE] {prim.GetPath()} extent={extent}")


def print_contact_api_status(prim_path: str):
    """打印 prim 的碰撞/接触相关 API 是否已挂载。"""
    prim = get_prim(prim_path)
    if not prim or not prim.IsValid():
        print(f"[WARN] prim not found: {prim_path}")
        return
    print(f"\n[INFO] Contact API status for {prim_path}")
    print(f"  RigidBodyAPI: {prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
    print(f"  CollisionAPI: {prim.HasAPI(UsdPhysics.CollisionAPI)}")
    print(f"  PhysxCollisionAPI: {prim.HasAPI(PhysxSchema.PhysxCollisionAPI)}")
    print(f"  PhysxContactReportAPI: {prim.HasAPI(PhysxSchema.PhysxContactReportAPI)}")


def ensure_cube_ready(cube_path: str):
    """确保方块具备碰撞/接触 API，以便附加触觉传感器。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        print(f"[WARN] 找不到方块 prim: {cube_path}")
        return
    updated = False
    if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(cube_prim)
        updated = True
    if not cube_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(cube_prim)
        updated = True
    if not cube_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(cube_prim)
        contact_report.CreateThresholdAttr(0.0)
        updated = True
    if updated:
        print(f"[INFO] 启用 {cube_path} 的碰撞/接触 API")


def ensure_contact_parent_ready(parent_path: str):
    """确保触觉传感器父节点具备碰撞/接触 API。"""
    prim = get_prim(parent_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"找不到传感器父节点: {parent_path}")
    updated = False
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        updated = True
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
        updated = True
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        PhysxSchema.PhysxCollisionAPI.Apply(prim)
        updated = True
    if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        contact_report.CreateThresholdAttr(0.0)
        updated = True
    if updated:
        print(f"[INFO] 启用 {parent_path} 的碰撞/接触 API")


def build_finger1_grid_offsets():
    """在 finger1_1 局部坐标系生成 Ny x Nz 网格偏移。"""
    offsets = []
    for i in range(FINGER1_GRID_NY):
        for j in range(FINGER1_GRID_NZ):
            y = FINGER1_Y_OFFSET + i * FINGER1_GRID_STEP_Y
            z = FINGER1_Z_OFFSET + j * FINGER1_GRID_STEP_Z
            offsets.append(Gf.Vec3d(FINGER1_X_OFFSET, y, z))
    return offsets


def attach_contact_sensors_on_finger1():
    """在 finger1_1 上按指定网格创建触觉传感器。"""
    ensure_contact_parent_ready(FINGER1_PARENT)
    clear_contact_sensors_under(FINGER1_PARENT, FINGER1_SENSOR_PREFIX)
    offsets = build_finger1_grid_offsets()
    sensor_paths = []
    for idx, offset in enumerate(offsets):
        sensor_name = f"{FINGER1_SENSOR_PREFIX}_{idx}"
        sensor_path = f"{FINGER1_PARENT}/{sensor_name}"
        success, _prim = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path=sensor_name,
            parent=FINGER1_PARENT,
            translation=offset,
            radius=CONTACT_SENSOR_RADIUS,
            min_threshold=CONTACT_SENSOR_MIN_THRESHOLD,
            max_threshold=100000.0,
            sensor_period=0.0,
        )
        if not success:
            print(f"[WARN] 创建触觉传感器失败: {sensor_path}")
            continue
        sensor_paths.append(sensor_path)
    print(f"[INFO] 已在 {FINGER1_PARENT} 创建 {len(sensor_paths)} 个传感器")
    return sensor_paths

def ensure_rigidbody_enabled(cube_path: str, mass: float = 0.2, enable_gravity=True, kinematic=None):
    """确保方块是可参与碰撞的刚体。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        print(f"[WARN] 找不到方块 prim: {cube_path}")
        return
    updated = False
    if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(cube_prim)
        updated = True
    if not cube_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
        mass_api.CreateMassAttr(mass)
        updated = True
    if not cube_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(cube_prim)
    else:
        physx_body = PhysxSchema.PhysxRigidBodyAPI(cube_prim)
    # 确保属性存在并写入期望值
    disable_g_attr = physx_body.GetDisableGravityAttr()
    if not disable_g_attr:
        disable_g_attr = physx_body.CreateDisableGravityAttr()
    disable_g_attr.Set(not enable_gravity)
    if kinematic is not None:
        try:
            kinematic_attr = physx_body.GetKinematicEnabledAttr()
            if not kinematic_attr:
                kinematic_attr = physx_body.CreateKinematicEnabledAttr()
            kinematic_attr.Set(bool(kinematic))
        except Exception as exc:
            print(f"[WARN] 设置 {cube_path} kinematic 失败: {exc}")
        else:
            updated = True
    else:
        updated = True
    if updated:
        print(f"[INFO] 启用 {cube_path} 的刚体/质量/重力 (gravity={enable_gravity}, kinematic={kinematic})")


def ensure_physics_scene():
    """确保存在 Physics Scene，并抓取重力方向/大小。"""
    global physics_scene, gravity_dir, gravity_mag
    stage_default_root = stage.GetDefaultPrim().GetPath().pathString if stage and stage.GetDefaultPrim().IsValid() else "/World"
    candidate_paths = [f"{stage_default_root}/physicsScene", "/World/physicsScene", "/physicsScene"]
    scene_prim = None
    for path in candidate_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.IsA(UsdPhysics.Scene):
            scene_prim = prim
            break
    if scene_prim is None:
        scene_path = candidate_paths[0]
        scene_prim = UsdPhysics.Scene.Define(stage, scene_path).GetPrim()
        print(f"[INFO] 创建 Physics Scene: {scene_path}")
    scene = UsdPhysics.Scene(scene_prim)
    # 若原有值无效则重置为默认重力
    dir_attr = scene.GetGravityDirectionAttr()
    mag_attr = scene.GetGravityMagnitudeAttr()
    dir_val = dir_attr.Get() if dir_attr else None
    mag_val = mag_attr.Get() if mag_attr else None
    dir_vec = Gf.Vec3d(dir_val[0], dir_val[1], dir_val[2]) if dir_val is not None else Gf.Vec3d(0.0, 0.0, -1.0)
    dir_len = float(dir_vec.GetLength())
    if dir_len < 1e-4 or not math.isfinite(dir_len):
        dir_vec = Gf.Vec3d(0.0, 0.0, -1.0)
    mag_value = float(mag_val) if mag_val is not None else 981.0
    if mag_value <= 0 or not math.isfinite(mag_value):
        mag_value = 981.0
    # 写回规范值
    (dir_attr if dir_attr else scene.CreateGravityDirectionAttr()).Set(Gf.Vec3f(dir_vec[0], dir_vec[1], dir_vec[2]))
    (mag_attr if mag_attr else scene.CreateGravityMagnitudeAttr()).Set(mag_value)
    gravity_dir = dir_vec
    gravity_mag = mag_value
    physics_scene = scene
    print(f"[INFO] Physics Scene gravity dir={gravity_dir} mag={gravity_mag}")
    return scene


def place_cube_above_other(top_cube: str, bottom_cube: str, offset: float = 0.08):
    """把 top_cube 放到 bottom_cube 的重力反方向上方一点，播放后会下落产生接触。"""
    top_prim = get_prim(top_cube)
    bottom_prim = get_prim(bottom_cube)
    if not (top_prim and bottom_prim and top_prim.IsValid() and bottom_prim.IsValid()):
        print(f"[WARN] 无法摆放方块，缺少 prim: {top_cube} 或 {bottom_cube}")
        return
    try:
        bottom_pos = UsdGeom.XformCommonAPI(bottom_prim).GetTranslateAttr().Get() or Gf.Vec3d(0.0, 0.0, 0.0)
        g = gravity_dir
        g_len = float(g.GetLength())
        if g_len < 1e-4 or not math.isfinite(g_len):
            g = Gf.Vec3d(0.0, 0.0, -1.0)
            g_len = 1.0
        up_vec = (-g / g_len) * offset
        new_pos = Gf.Vec3d(bottom_pos[0] + up_vec[0], bottom_pos[1] + up_vec[1], bottom_pos[2] + up_vec[2])
        UsdGeom.XformCommonAPI(top_prim).SetTranslate(new_pos)
        print(f"[INFO] 将 {top_cube} 放到 {bottom_cube} 上方 {offset}m 处 (沿重力反方向)，位置 {new_pos}")
    except Exception as exc:
        print(f"[WARN] 摆放方块失败: {exc}")


def _compute_positive_y_offset(prim, padding=0.005):
    """尝试用 extent 估计 Y 方向正面的偏移，失败则使用默认值。"""
    default_offset = Gf.Vec3d(0.0, 0.05, 0.0)
    try:
        boundable = UsdGeom.Boundable(prim)
        extent = boundable.ComputeExtent(Usd.TimeCode.Default())
        if extent and len(extent) == 2:
            min_pt, max_pt = extent
            return Gf.Vec3d(0.0, float(max_pt[1]) + padding, 0.0)
    except Exception:
        pass
    return default_offset


def _compute_negative_y_offset(prim, padding=0.005):
    """尝试用 extent 估计 Y 方向负面的偏移，失败则使用默认值。"""
    default_offset = Gf.Vec3d(0.0, -0.05, 0.0)
    try:
        boundable = UsdGeom.Boundable(prim)
        extent = boundable.ComputeExtent(Usd.TimeCode.Default())
        if extent and len(extent) == 2:
            min_pt, max_pt = extent
            return Gf.Vec3d(0.0, float(min_pt[1]) - padding, 0.0)
    except Exception:
        pass
    return default_offset


def _compute_face_offset(prim, direction, padding=0.005, fallback=0.05):
    """根据方向向量 (x,y,z) 估计面外的偏移，用 extent 失败时使用回退值。"""
    # direction 例如 (1,0,0) 表示 +X 面
    default_offset = Gf.Vec3d(direction[0] * fallback, direction[1] * fallback, direction[2] * fallback)
    try:
        boundable = UsdGeom.Boundable(prim)
        extent = boundable.ComputeExtent(Usd.TimeCode.Default())
        if not extent or len(extent) != 2:
            return default_offset
        min_pt, max_pt = extent
        offset = [0.0, 0.0, 0.0]
        # 找到主要轴向
        axis = max(range(3), key=lambda i: abs(direction[i]))
        if direction[axis] >= 0:
            offset[axis] = float(max_pt[axis]) + padding
        else:
            offset[axis] = float(min_pt[axis]) - padding
        return Gf.Vec3d(offset[0], offset[1], offset[2])
    except Exception:
        return default_offset


def _compute_face_grid_offsets(
    prim, direction, grid_size=GRID_SIZE, padding=GRID_PADDING, inset=GRID_INSET, fallback_extent=0.05
):
    """在指定面上生成 grid_size x grid_size 的偏移点（局部坐标）。"""
    try:
        boundable = UsdGeom.Boundable(prim)
        extent = boundable.ComputeExtent(Usd.TimeCode.Default())
    except Exception:
        extent = None

    if not extent or len(extent) != 2:
        half = fallback_extent * 0.5
        min_pt = Gf.Vec3d(-half, -half, -half)
        max_pt = Gf.Vec3d(half, half, half)
    else:
        min_pt, max_pt = extent
        min_pt = Gf.Vec3d(min_pt[0], min_pt[1], min_pt[2])
        max_pt = Gf.Vec3d(max_pt[0], max_pt[1], max_pt[2])

    axis = max(range(3), key=lambda i: abs(direction[i]))
    fixed = max_pt[axis] + padding if direction[axis] >= 0 else min_pt[axis] - padding

    axes = [0, 1, 2]
    axes.remove(axis)
    ranges = []
    for ax in axes:
        low = float(min_pt[ax]) + inset
        high = float(max_pt[ax]) - inset
        if high <= low:
            low = float(min_pt[ax])
            high = float(max_pt[ax])
        ranges.append(np.linspace(low, high, grid_size))

    offsets = []
    for r, v0 in enumerate(ranges[0]):
        for c, v1 in enumerate(ranges[1]):
            coords = [0.0, 0.0, 0.0]
            coords[axis] = float(fixed)
            coords[axes[0]] = float(v0)
            coords[axes[1]] = float(v1)
            offsets.append((r, c, Gf.Vec3d(coords[0], coords[1], coords[2])))
    return offsets


def _estimate_grid_radius(prim, direction, grid_size=GRID_SIZE, fallback_extent=0.05):
    """根据面尺寸估计每个传感器的半径（用于 grid 传感器）。"""
    try:
        boundable = UsdGeom.Boundable(prim)
        extent = boundable.ComputeExtent(Usd.TimeCode.Default())
    except Exception:
        extent = None

    if not extent or len(extent) != 2:
        span = fallback_extent
    else:
        min_pt, max_pt = extent
        axis = max(range(3), key=lambda i: abs(direction[i]))
        axes = [0, 1, 2]
        axes.remove(axis)
        span0 = float(max_pt[axes[0]]) - float(min_pt[axes[0]])
        span1 = float(max_pt[axes[1]]) - float(min_pt[axes[1]])
        span = min(span0, span1)

    cell = span / float(grid_size) if grid_size > 0 else span
    radius = max(0.001, cell * 0.45)
    return radius


def attach_contact_sensor_on_positive_y(cube_path: str):
    """在立方体的 +Y 方向附加一个接触传感器。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        print(f"[WARN] 跳过，找不到方块: {cube_path}")
        return
    ensure_cube_ready(cube_path)
    offset = _compute_positive_y_offset(cube_prim)
    sensor_path = f"{cube_path}/{CONTACT_SENSOR_NAME}"
    success, prim = omni.kit.commands.execute(
        "IsaacSensorCreateContactSensor",
        path=CONTACT_SENSOR_NAME,
        parent=cube_path,
        translation=offset,
        radius=CONTACT_SENSOR_RADIUS,
        min_threshold=CONTACT_SENSOR_MIN_THRESHOLD,
        max_threshold=100000.0,
        sensor_period=0.0,
    )
    if not success:
        print(f"[WARN] 创建接触传感器失败: {sensor_path}")
        return
    print(f"[INFO] 已在 {cube_path} 正 Y 方向附加传感器 {sensor_path}，偏移 {offset}")
    try:
        # 写回局部平移，防止 op 顺序问题
        prim_obj = get_prim(sensor_path)
        if prim_obj and prim_obj.IsValid():
            xformable = UsdGeom.Xformable(prim_obj)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(offset[0], offset[1], offset[2]))
    except Exception as exc:
        print(f"[WARN] 无法写入传感器偏移: {exc}")
    # 创建读取接口，方便后续扩展（当前仅初始化）
    return sensor_path


def attach_contact_sensor_on_negative_y(cube_path: str, sensor_name: str = CONTACT_SENSOR_NAME_NEG_Y):
    """在立方体的 -Y 方向附加一个接触传感器。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        print(f"[WARN] 跳过，找不到方块: {cube_path}")
        return
    ensure_cube_ready(cube_path)
    offset = _compute_negative_y_offset(cube_prim)
    sensor_path = f"{cube_path}/{sensor_name}"
    success, prim = omni.kit.commands.execute(
        "IsaacSensorCreateContactSensor",
        path=sensor_name,
        parent=cube_path,
        translation=offset,
        radius=CONTACT_SENSOR_RADIUS,
        min_threshold=CONTACT_SENSOR_MIN_THRESHOLD,
        max_threshold=100000.0,
        sensor_period=0.0,
    )
    if not success:
        print(f"[WARN] 创建接触传感器失败: {sensor_path}")
        return
    print(f"[INFO] 已在 {cube_path} 负 Y 方向附加传感器 {sensor_path}，偏移 {offset}")
    try:
        prim_obj = get_prim(sensor_path)
        if prim_obj and prim_obj.IsValid():
            xformable = UsdGeom.Xformable(prim_obj)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(offset[0], offset[1], offset[2]))
    except Exception as exc:
        print(f"[WARN] 无法写入传感器偏移: {exc}")
    return sensor_path


def attach_contact_sensor_on_face(cube_path: str, face_label: str, direction, name_prefix: str):
    """在指定面附加传感器，face_label 为 'pos_x' 等，direction 为对应向量。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        print(f"[WARN] 跳过，找不到方块: {cube_path}")
        return None
    ensure_cube_ready(cube_path)
    offset = _compute_face_offset(cube_prim, direction)
    sensor_name = f"{name_prefix}_{face_label}"
    sensor_path = f"{cube_path}/{sensor_name}"
    success, prim = omni.kit.commands.execute(
        "IsaacSensorCreateContactSensor",
        path=sensor_name,
        parent=cube_path,
        translation=offset,
        radius=CONTACT_SENSOR_RADIUS,
        min_threshold=CONTACT_SENSOR_MIN_THRESHOLD,
        max_threshold=100000.0,
        sensor_period=0.0,
    )
    if not success:
        print(f"[WARN] 创建接触传感器失败: {sensor_path}")
        return None
    print(f"[INFO] 已在 {cube_path} 的 {face_label} 面附加传感器 {sensor_path}，偏移 {offset}")
    try:
        prim_obj = get_prim(sensor_path)
        if prim_obj and prim_obj.IsValid():
            print(f"[INFO] contact sensor prim {sensor_path} is_valid=True")
            xformable = UsdGeom.Xformable(prim_obj)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(offset[0], offset[1], offset[2]))
        else:
            print(f"[INFO] contact sensor prim {sensor_path} is_valid=False")
    except Exception as exc:
        print(f"[WARN] 无法写入传感器偏移: {exc}")
    return sensor_path


def attach_contact_sensors_grid_on_face(cube_path: str, face_label: str, direction, name_prefix: str, grid_size=GRID_SIZE):
    """在指定面上创建 grid_size x grid_size 触觉传感器。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        print(f"[WARN] 跳过，找不到方块: {cube_path}")
        return []
    ensure_cube_ready(cube_path)
    print(f"[INFO] 开始创建 {cube_path} 的 {face_label} 面传感器网格...")
    offsets = _compute_face_grid_offsets(cube_prim, direction, grid_size=grid_size)
    radius = CONTACT_SENSOR_RADIUS if CONTACT_SENSOR_RADIUS > 0 else _estimate_grid_radius(cube_prim, direction, grid_size)
    sensor_paths = []
    for r, c, offset in offsets:
        sensor_name = f"{name_prefix}_{face_label}_{r}_{c}"
        sensor_path = f"{cube_path}/{sensor_name}"
        success, prim = omni.kit.commands.execute(
        "IsaacSensorCreateContactSensor",
        path=sensor_name,
        parent=cube_path,
        translation=offset,
        radius=radius,
        min_threshold=CONTACT_SENSOR_MIN_THRESHOLD,
        max_threshold=100000.0,
        sensor_period=0.0,
    )
        if not success:
            print(f"[WARN] 创建接触传感器失败: {sensor_path}")
            continue
        try:
            prim_obj = get_prim(sensor_path)
            if prim_obj and prim_obj.IsValid():
                xformable = UsdGeom.Xformable(prim_obj)
                xformable.ClearXformOpOrder()
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(offset[0], offset[1], offset[2]))
        except Exception as exc:
            print(f"[WARN] 无法写入传感器偏移: {exc}")
        sensor_paths.append(sensor_path)
    print(f"[INFO] 已在 {cube_path} 的 {face_label} 面创建 {len(sensor_paths)} 个传感器 (radius={radius:.4f})")
    return sensor_paths


def attach_contact_sensors_selected_faces(cube_path: str, name_prefix: str, face_labels):
    """为指定面挂上传感器网格，返回 {face_label: sensors}。"""
    clear_contact_sensors_under(cube_path, name_prefix)
    sensors = {}
    for face_label in face_labels:
        direction = FACE_DIRECTIONS.get(face_label)
        if direction is None:
            print(f"[WARN] 未知面标签: {face_label}")
            continue
        sensor_list = attach_contact_sensors_grid_on_face(cube_path, face_label, direction, name_prefix)
        sensors[face_label] = sensor_list
    return sensors


def clear_contact_sensors_under(cube_path: str, name_prefix: str):
    """清理指定 cube 下已存在的 ContactSensor，并隐藏可视化球体。"""
    cube_prim = get_prim(cube_path)
    if not cube_prim or not cube_prim.IsValid():
        return
    removed = 0
    for prim in list(stage.Traverse()):
        path_str = prim.GetPath().pathString
        if not path_str.startswith(cube_path + "/"):
            continue
        if prim.GetTypeName() == "IsaacContactSensor":
            stage.RemovePrim(path_str)
            sensor_viz_map.pop(path_str, None)
            removed += 1
            continue
        name = prim.GetName()
        if name.startswith(name_prefix) and name.endswith("_viz"):
            imageable = UsdGeom.Imageable(prim)
            imageable.GetVisibilityAttr().Set("invisible")
            sensor_viz_map.pop(path_str, None)
            removed += 1
    if removed:
        print(f"[INFO] 清理 {cube_path} 下旧触觉传感器/可视化 {removed} 个")


def clear_all_sensor_visuals(root_path: str):
    """隐藏所有 *_viz 的可视化球体，避免访问已删除的 prim。"""
    root = get_prim(root_path)
    if not root or not root.IsValid():
        return
    removed = 0
    for p in list(stage.Traverse()):
        path_str = p.GetPath().pathString
        if not path_str.startswith(root_path + "/"):
            continue
        name = path_str.rsplit("/", 1)[-1]
        if name.endswith("_viz"):
            imageable = UsdGeom.Imageable(p)
            imageable.GetVisibilityAttr().Set("invisible")
            removed += 1
    if removed:
        sensor_viz_map.clear()
        print(f"[INFO] 清理 {root_path} 下所有可视化球体 {removed} 个")


def clear_blocks_under(parent_path: str, block_prefix: str, sensor_prefix: str):
    """清理指定 parent 下的方块与其传感器/可视化。"""
    prim = get_prim(parent_path)
    if not prim or not prim.IsValid():
        return
    removed = 0
    for p in list(stage.Traverse()):
        path_str = p.GetPath().pathString
        if not path_str.startswith(parent_path + "/"):
            continue
        name = path_str.rsplit("/", 1)[-1]
        if (
            name.startswith(sensor_prefix)
            or name.startswith(OLD_CUBE1_POSZ_CYL_PREFIX)
        ):
            stage.RemovePrim(path_str)
            sensor_viz_map.pop(path_str, None)
            removed += 1
        elif name.startswith(block_prefix):
            imageable = UsdGeom.Imageable(p)
            imageable.GetVisibilityAttr().Set("invisible")
            removed += 1
        elif name.endswith("_viz") and "PosZ" in name:
            imageable = UsdGeom.Imageable(p)
            imageable.GetVisibilityAttr().Set("invisible")
            sensor_viz_map.pop(path_str, None)
            removed += 1
    if removed:
        print(f"[INFO] 清理 {parent_path} 下旧方块/传感器 {removed} 个")


def build_cube1_posz_offsets():
    """基于 Cube1 的 pos_z 面生成方块网格偏移。"""
    prim = get_prim(CUBE1_POSZ_PARENT)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"找不到 cube1 prim: {CUBE1_POSZ_PARENT}")
    direction = FACE_DIRECTIONS["pos_z"]
    rows, cols = CUBE1_POSZ_BLOCK_GRID
    offsets = _compute_face_grid_offsets(prim, direction, grid_size=rows)
    # _compute_face_grid_offsets 返回 rows x rows，需裁成 rows x cols
    trimmed = []
    for r in range(rows):
        for c in range(cols):
            idx = r * rows + c
            if idx < len(offsets):
                trimmed.append(offsets[idx])
    return trimmed


def attach_blocks_with_sensors_on_cube1_posz():
    """在 Cube1 的 pos_z 面创建小方块，并在方块顶部放 Contact Sensor。"""
    ensure_cube_ready(CUBE1_POSZ_PARENT)
    clear_blocks_under(CUBE1_POSZ_PARENT, CUBE1_POSZ_BLOCK_PREFIX, CUBE1_POSZ_SENSOR_PREFIX)
    prim = get_prim(CUBE1_POSZ_PARENT)
    offsets = build_cube1_posz_offsets()
    sensor_paths = []
    for idx, (_r, _c, face_offset) in enumerate(offsets):
        block_name = f"{CUBE1_POSZ_BLOCK_PREFIX}_{idx}"
        block_path = f"{CUBE1_POSZ_PARENT}/{block_name}"
        cube = UsdGeom.Cube.Define(stage, block_path)
        cube.CreateSizeAttr(CUBE1_POSZ_BLOCK_SIZE)
        cube.CreateDisplayColorAttr().Set([CUBE1_POSZ_BLOCK_COLOR])
        UsdGeom.Imageable(cube).GetVisibilityAttr().Set("inherited")
        # 给方块添加碰撞
        block_prim = get_prim(block_path)
        if block_prim and block_prim.IsValid():
            if not block_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(block_prim)
            if not block_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(block_prim)
        # 将方块中心抬到面外
        block_center = Gf.Vec3d(
            face_offset[0],
            face_offset[1],
            face_offset[2] + (CUBE1_POSZ_BLOCK_SIZE * 0.5 + CUBE1_POSZ_BLOCK_PADDING),
        )
        UsdGeom.XformCommonAPI(cube).SetTranslate(block_center)

        sensor_name = f"{CUBE1_POSZ_SENSOR_PREFIX}_{idx}"
        sensor_path = f"{block_path}/{sensor_name}"
        success, _prim = omni.kit.commands.execute(
            "IsaacSensorCreateContactSensor",
            path=sensor_name,
            parent=block_path,
            translation=Gf.Vec3d(0.0, 0.0, CUBE1_POSZ_BLOCK_SIZE * 0.5),
            radius=CONTACT_SENSOR_RADIUS,
            min_threshold=CONTACT_SENSOR_MIN_THRESHOLD,
            max_threshold=100000.0,
            sensor_period=0.0,
        )
        if not success:
            print(f"[WARN] 创建方块触觉传感器失败: {sensor_path}")
            continue
        sensor_paths.append(sensor_path)
    print(f"[INFO] 已在 Cube1 pos_z 面创建 {len(sensor_paths)} 个方块传感器")
    return sensor_paths


def _build_contact_ui_window(title: str, face_labels, grid_size=GRID_SIZE):
    """创建 contact 状态窗口，返回 (window, label_map)。"""
    window = omni.ui.Window(title, width=420, height=720)
    label_map = {}
    with window.frame:
        with omni.ui.VStack(spacing=6):
            for face_label in face_labels:
                with omni.ui.CollapsableFrame(face_label, collapsed=False):
                    with omni.ui.VGrid(row_height=18, column_width=18):
                        labels = []
                        for _r in range(grid_size):
                            for _c in range(grid_size):
                                labels.append(
                                    omni.ui.Rectangle(
                                        width=12,
                                        height=12,
                                        style={"background_color": omni.ui.color(*VIZ_COLOR_OPEN)},
                                    )
                                )
                        label_map[face_label] = labels
    return window, label_map


def _update_contact_ui(label_map, sensors_by_face):
    """根据接触状态更新 UI（numContacts>0 视为接触）。"""
    for face_label, sensor_paths in sensors_by_face.items():
        labels = label_map.get(face_label, [])
        for idx, sensor_path in enumerate(sensor_paths):
            if idx >= len(labels):
                break
            reading = read_contact_sensor_reading(sensor_path)
            in_contact = bool(reading.in_contact) if reading.is_valid else False
            labels[idx].style = {
                "background_color": omni.ui.color(*(VIZ_COLOR_CONTACT if in_contact else VIZ_COLOR_OPEN))
            }


def _get_local_translation(prim_path: str):
    prim = get_prim(prim_path)
    if not prim:
        return Gf.Vec3d(0.0, 0.0, 0.0)
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op.Get()
    return Gf.Vec3d(0.0, 0.0, 0.0)


def _set_local_translation(prim_path: str, translation: Gf.Vec3d):
    prim = get_prim(prim_path)
    if not prim:
        return False
    UsdGeom.XformCommonAPI(prim).SetTranslate(translation)
    return True


def _local_to_world_point(prim, local_point):
    xformable = UsdGeom.Xformable(prim)
    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return xform.Transform(local_point)


def _local_to_world_dir(prim, local_dir):
    xformable = UsdGeom.Xformable(prim)
    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation = xform.ExtractRotation()
    return rotation.TransformDir(local_dir)


def build_raycast_grid_offsets(prim_path: str, face_label: str, grid_size: int, padding: float):
    prim = get_prim(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"找不到 prim: {prim_path}")
    direction = FACE_DIRECTIONS.get(face_label)
    if direction is None:
        raise RuntimeError(f"未知面标签: {face_label}")
    return _compute_face_grid_offsets(prim, direction, grid_size=grid_size, padding=padding)


def build_raycast_grid_offsets_rect(prim_path: str, face_label: str, rows: int, cols: int, padding: float):
    prim = get_prim(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"找不到 prim: {prim_path}")
    direction = FACE_DIRECTIONS.get(face_label)
    if direction is None:
        raise RuntimeError(f"未知面标签: {face_label}")
    offsets = _compute_face_grid_offsets(prim, direction, grid_size=rows, padding=padding)
    trimmed = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < len(offsets):
                trimmed.append(offsets[idx])
    return trimmed


def _distance_to_color(distance, max_distance):
    if distance is None:
        return RAYCAST_COLOR_NO_HIT
    t = min(max(distance / max_distance, 0.0), 1.0)
    r = 0.9 * t + 0.1
    g = 0.9 * (1.0 - t) + 0.1
    b = 0.2
    return (r, g, b)


def _query_raycast_distances(prim_path, offsets, face_label, max_distance):
    prim = get_prim(prim_path)
    if not prim or not prim.IsValid():
        return []
    local_dir = FACE_DIRECTIONS.get(face_label)
    if local_dir is None:
        return []
    world_dir = _local_to_world_dir(prim, Gf.Vec3d(*local_dir))
    dir_tuple = (float(world_dir[0]), float(world_dir[1]), float(world_dir[2]))
    distances = []
    for _r, _c, local_offset in offsets:
        world_pt = _local_to_world_point(prim, local_offset)
        origin = (float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))
        hit = physx_scene_query.raycast_closest(origin, dir_tuple, max_distance)
        if hit and hit.get("hit"):
            if hit.get("rigidBody") == prim_path:
                distances.append(None)
            else:
                distances.append(float(hit.get("distance", 0.0)))
        else:
            distances.append(None)
    return distances


def _build_raycast_ui_window(title: str, rows: int, cols: int, show_values: bool = True):
    window = omni.ui.Window(title, width=max(200, cols * 28 + 60), height=max(200, rows * 28 + 60))
    cells = []
    with window.frame:
        with omni.ui.VGrid(row_height=28, column_width=28):
            for _r in range(rows):
                for _c in range(cols):
                    with omni.ui.ZStack():
                        rect = omni.ui.Rectangle(
                            width=22,
                            height=22,
                            style={"background_color": omni.ui.color(*RAYCAST_COLOR_NO_HIT)},
                        )
                        label = omni.ui.Label(
                            RAYCAST_UI_EMPTY_TEXT,
                            alignment=omni.ui.Alignment.CENTER,
                            style={
                                "font_size": RAYCAST_UI_FONT_SIZE,
                                "color": omni.ui.color(*RAYCAST_UI_TEXT_COLOR),
                            },
                            visible=show_values,
                        )
                    cells.append({"rect": rect, "label": label})
    return window, cells


def _update_raycast_ui(cells, distances, max_distance):
    for idx, dist in enumerate(distances):
        if idx >= len(cells):
            break
        color = _distance_to_color(dist, max_distance)
        cells[idx]["rect"].style = {"background_color": omni.ui.color(*color)}
        if dist is None:
            cells[idx]["label"].text = RAYCAST_UI_EMPTY_TEXT
        else:
            cells[idx]["label"].text = RAYCAST_UI_VALUE_FORMAT.format(dist)


def _create_raycast_origin_visuals(parent_prim_path, offsets, radius, color, prefix="ray_origin"):
    parent_prim = get_prim(parent_prim_path)
    if not parent_prim or not parent_prim.IsValid():
        print(f"[WARN] 找不到 prim: {parent_prim_path}")
        return []
    if not get_prim(FINGER1_RAYCAST_ORIGIN_PARENT):
        UsdGeom.Xform.Define(stage, FINGER1_RAYCAST_ORIGIN_PARENT)
    created = []
    for r, c, local_offset in offsets:
        sphere_path = f"{FINGER1_RAYCAST_ORIGIN_PARENT}/{prefix}_r{r}_c{c}"
        sphere = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere.GetRadiusAttr().Set(radius)
        sphere.CreateDisplayColorAttr().Set([color])
        UsdGeom.XformCommonAPI(sphere).SetTranslate(local_offset)
        created.append(sphere_path)
    return created

def _create_sensor_visuals(sensors_by_face, radius):
    """为每个传感器创建可视化球体。"""
    for face_label, sensor_paths in sensors_by_face.items():
        for sensor_path in sensor_paths:
            sensor_name = os.path.basename(sensor_path)
            parent_path = str(Sdf.Path(sensor_path).GetParentPath())
            viz_path = f"{parent_path}/{sensor_name}_viz"
            if get_prim(viz_path):
                sensor_viz_map[sensor_path] = viz_path
                continue
            sphere = UsdGeom.Sphere.Define(stage, viz_path)
            sphere.GetRadiusAttr().Set(radius * VIZ_RADIUS_SCALE)
            sphere.CreateDisplayColorAttr().Set([VIZ_COLOR_OPEN])
            offset = _get_local_translation(sensor_path)
            UsdGeom.XformCommonAPI(sphere).SetTranslate(offset)
            sensor_viz_map[sensor_path] = viz_path


def _update_sensor_visuals(sensors_by_face):
    """更新可视化球体颜色，接触=绿，未接触=红。"""
    for sensor_paths in sensors_by_face.values():
        for sensor_path in sensor_paths:
            viz_path = sensor_viz_map.get(sensor_path)
            if not viz_path:
                continue
            viz_prim = get_prim(viz_path)
            if not viz_prim or not viz_prim.IsValid():
                sensor_viz_map.pop(sensor_path, None)
                continue
            reading = read_contact_sensor_reading(sensor_path)
            color = VIZ_COLOR_CONTACT if reading.in_contact else VIZ_COLOR_OPEN
            try:
                geom = UsdGeom.Gprim(viz_prim)
                geom.CreateDisplayColorAttr().Set([color])
            except Exception:
                continue


def _record_force_series(sensor_paths, series_map, min_force):
    """记录传感器力-时间序列，仅保存 force>min_force 的点。"""
    for sensor_path in sensor_paths:
        reading = read_contact_sensor_reading(sensor_path)
        if not reading.is_valid:
            continue
        if reading.value <= min_force:
            continue
        series_map.setdefault(sensor_path, []).append((reading.time, reading.value))


def _record_force_history(sensor_paths, history_map, history_len):
    """记录传感器力历史，用于实时曲线。"""
    for sensor_path in sensor_paths:
        reading = read_contact_sensor_reading(sensor_path)
        if not reading.is_valid:
            continue
        history = history_map.setdefault(sensor_path, [])
        history.append(float(reading.value))
        if len(history) > history_len:
            del history[: len(history) - history_len]


def _record_contact_series(sensor_paths, series_map):
    """记录传感器接触-时间序列，保存 0/1。"""
    for sensor_path in sensor_paths:
        reading = read_contact_sensor_reading(sensor_path)
        if not reading.is_valid:
            continue
        series_map.setdefault(sensor_path, []).append((reading.time, 1 if reading.in_contact else 0))


def _plot_force_series_by_row(sensor_paths, series_map, grid_size, output_path):
    """将 pos_y 传感器的力-时间曲线按行绘制到一张图中。"""
    if not sensor_paths:
        print("[WARN] pos_y 传感器为空，无法绘图。")
        return
    rows = grid_size
    row_has_data = [False] * rows
    for idx, sensor_path in enumerate(sensor_paths):
        row = idx // grid_size
        if row >= rows:
            break
        if series_map.get(sensor_path):
            row_has_data[row] = True
    rows_to_plot = [r for r, has in enumerate(row_has_data) if has]
    if not rows_to_plot:
        print("[WARN] pos_y 力-时间无有效数据，跳过绘图。")
        return
    fig, axes = plt.subplots(
        len(rows_to_plot), 1, figsize=(10, max(2, len(rows_to_plot) * 1.6)), sharex=True
    )
    if len(rows_to_plot) == 1:
        axes = [axes]
    row_to_ax = {row: ax_idx for ax_idx, row in enumerate(rows_to_plot)}
    for idx, sensor_path in enumerate(sensor_paths):
        row = idx // grid_size
        if row >= rows:
            break
        if row not in row_to_ax:
            continue
        data = series_map.get(sensor_path)
        if not data:
            continue
        times = [p[0] for p in data]
        values = [p[1] for p in data]
        axes[row_to_ax[row]].plot(times, values, linewidth=1.0)
    for row in rows_to_plot:
        ax = axes[row_to_ax[row]]
        ax.set_ylabel(f"row {row}")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Cube1 pos_y force-time (non-zero)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 已保存 pos_y 力-时间曲线图: {output_path}")


def _plot_contact_series_by_row(sensor_paths, series_map, grid_size, output_path):
    """将 pos_y 传感器的接触(0/1)-时间曲线按行绘制到一张图中。"""
    if not sensor_paths:
        print("[WARN] pos_y 传感器为空，无法绘图。")
        return
    rows = grid_size
    row_has_data = [False] * rows
    for idx, sensor_path in enumerate(sensor_paths):
        row = idx // grid_size
        if row >= rows:
            break
        if series_map.get(sensor_path):
            row_has_data[row] = True
    rows_to_plot = [r for r, has in enumerate(row_has_data) if has]
    if not rows_to_plot:
        print("[WARN] pos_y 接触-时间无有效数据，跳过绘图。")
        return
    fig, axes = plt.subplots(
        len(rows_to_plot), 1, figsize=(10, max(2, len(rows_to_plot) * 1.6)), sharex=True
    )
    if len(rows_to_plot) == 1:
        axes = [axes]
    row_to_ax = {row: ax_idx for ax_idx, row in enumerate(rows_to_plot)}
    for idx, sensor_path in enumerate(sensor_paths):
        row = idx // grid_size
        if row >= rows:
            break
        if row not in row_to_ax:
            continue
        data = series_map.get(sensor_path)
        if not data:
            continue
        times = [p[0] for p in data]
        values = [p[1] for p in data]
        axes[row_to_ax[row]].step(times, values, where="post", linewidth=1.0)
    for row in rows_to_plot:
        ax = axes[row_to_ax[row]]
        ax.set_ylabel(f"row {row}")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Cube1 pos_y contact (0/1)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 已保存 pos_y 接触-时间曲线图: {output_path}")


def _plot_contact_force_combined(sensor_paths, contact_series, force_series, grid_size, output_path):
    """将 pos_y 的接触(左)与力(右)按行拼成一张图。"""
    if not sensor_paths:
        print("[WARN] pos_y 传感器为空，无法绘图。")
        return
    rows = grid_size
    row_has_data = [False] * rows
    for idx, sensor_path in enumerate(sensor_paths):
        row = idx // grid_size
        if row >= rows:
            break
        if contact_series.get(sensor_path) or force_series.get(sensor_path):
            row_has_data[row] = True
    rows_to_plot = [r for r, has in enumerate(row_has_data) if has]
    if not rows_to_plot:
        print("[WARN] pos_y 无有效数据，跳过拼图。")
        return
    fig, axes = plt.subplots(
        len(rows_to_plot), 2, figsize=(12, max(2, len(rows_to_plot) * 1.6)), sharex=True
    )
    if len(rows_to_plot) == 1:
        axes = [axes]
    row_to_ax = {row: ax_idx for ax_idx, row in enumerate(rows_to_plot)}
    for idx, sensor_path in enumerate(sensor_paths):
        row = idx // grid_size
        if row >= rows:
            break
        if row not in row_to_ax:
            continue
        ax_row = axes[row_to_ax[row]]
        contact_data = contact_series.get(sensor_path)
        if contact_data:
            times = [p[0] for p in contact_data]
            values = [p[1] for p in contact_data]
            ax_row[0].step(times, values, where="post", linewidth=1.0)
        force_data = force_series.get(sensor_path)
        if force_data:
            times = [p[0] for p in force_data]
            values = [p[1] for p in force_data]
            ax_row[1].plot(times, values, linewidth=1.0)
    for row in rows_to_plot:
        ax_row = axes[row_to_ax[row]]
        ax_row[0].set_ylabel(f"row {row}")
        ax_row[0].set_ylim(-0.1, 1.1)
        ax_row[0].grid(True, alpha=0.2)
        ax_row[1].grid(True, alpha=0.2)
    axes[-1][0].set_xlabel("time (s)")
    axes[-1][1].set_xlabel("time (s)")
    axes[0][0].set_title("contact (0/1)")
    axes[0][1].set_title("force")
    fig.suptitle("Cube1 pos_y contact (left) + force (right)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] 已保存 pos_y 接触+力 拼图: {output_path}")


def _build_force_plot_window(title: str, grid_size: int, sensor_paths):
    """创建实时力曲线窗口，返回 curve_map。"""
    window = omni.ui.Window(title, width=520, height=360)
    curve_map = {}
    colors = [
        (0.2, 0.9, 0.2, 1.0),
        (0.2, 0.6, 0.95, 1.0),
        (0.95, 0.6, 0.2, 1.0),
        (0.9, 0.2, 0.6, 1.0),
    ]
    with window.frame:
        with omni.ui.VStack(spacing=6):
            for row in range(grid_size):
                row_start = row * grid_size
                row_end = row_start + grid_size
                row_sensors = sensor_paths[row_start:row_end]
                if not row_sensors:
                    continue
                with omni.ui.HStack(height=80):
                    omni.ui.Label(f"row {row}", width=50)
                    scene_view = sc.SceneView(
                        aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT, height=70
                    )
                    with scene_view.scene:
                        for idx, sensor_path in enumerate(row_sensors):
                            color = colors[idx % len(colors)]
                            curve = sc.Curve(
                                [[-1, -1, 0], [1, -1, 0]],
                                colors=[color, color],
                                thicknesses=[2.0, 2.0],
                                curve_type=sc.Curve.CurveType.LINEAR,
                            )
                            curve_map[sensor_path] = curve
    return window, curve_map


def _update_force_plot(curve_map, history_map, history_len, y_max):
    """刷新实时力曲线。"""
    if not curve_map:
        return
    x_values = []
    if history_len <= 1:
        x_values = [0.0]
    else:
        for i in range(history_len):
            x_values.append(-1.0 + 2.0 * (i / (history_len - 1)))
    for sensor_path, curve in curve_map.items():
        history = history_map.get(sensor_path)
        if not history:
            continue
        values = history[-history_len:]
        if len(values) < history_len:
            pad = [0.0] * (history_len - len(values))
            values = pad + values
        positions = []
        for x, v in zip(x_values, values):
            y = -1.0 + 2.0 * min(max(v / max(y_max, 1e-6), 0.0), 1.0)
            positions.append([x, y, 0])
        curve.positions = positions


def log_contact_sensor_groups(sensor_groups):
    """打印多个组（cube）的传感器读数。sensor_groups 为 [(name, sensors_dict), ...]。"""
    results = {}
    timestamp = time.time() - initial_time if initial_time is not None else None
    if timestamp is not None:
        print(f"[TIME] {timestamp:.3f}s since start")
    for group_name, sensors in sensor_groups:
        group_result = {}
        if not sensors:
            results[group_name] = group_result
            continue
        for face_label, sensor_path in sensors.items():
            if isinstance(sensor_path, (list, tuple)):
                contact_count = 0
                for sp in sensor_path:
                    reading = read_contact_sensor_reading(sp)
                    contact_count += int(reading.in_contact) if reading.is_valid else 0
                group_result[face_label] = contact_count
                print(f"[CONTACT][{group_name}][{face_label}] contact_count={contact_count}/{len(sensor_path)}")
            else:
                reading = read_contact_sensor_reading(sensor_path)
                group_result[face_label] = reading
                if reading.is_valid:
                    print(
                        f"[CONTACT][{group_name}][{face_label}] "
                        f"is_valid={reading.is_valid} time={reading.time:.3f} "
                        f"value={reading.value:.4f} in_contact={reading.in_contact}"
                    )
        results[group_name] = group_result
    return results


def read_contact_sensor_reading(sensor_path: str):
    """读取接触传感器并返回标准化读数。"""
    timestamp = time.time() - initial_time if initial_time is not None else time.time()
    if not sensor_path:
        return CsSensorReading(False, timestamp, 0.0, False)
    reading = contact_sensor_iface.get_sensor_reading(sensor_path, True)
    is_valid = bool(reading.is_valid)
    in_contact = bool(reading.in_contact)
    value = float(reading.value)
    return CsSensorReading(is_valid, timestamp, value, in_contact)


def print_contact_sensor_reading(sensor_path: str):
    """读取并打印接触传感器的当前帧摘要，同时返回数据。"""
    global initial_time
    reading = read_contact_sensor_reading(sensor_path)
    if reading.is_valid:
        print(
            f"[CONTACT] is_valid={reading.is_valid} time={reading.time:.3f} "
            f"value={reading.value:.4f} in_contact={reading.in_contact}"
        )
    return reading


def main():
    global stage, contact_sensor, contact_sensor_cube2_neg_y, cube1_sensors, cube2_sensors, initial_time
    print("[INFO] main 开始，准备加载场景与传感器")
    try:
        stage = open_stage()
        print("stage opened")
        ensure_physics_scene()
        clear_all_sensor_visuals("/ur3")
        print_prim_pose_and_extent([
            "/ur3/umi",
            "/ur3/umi/finger1_1",
            "/ur3/umi/finger2_1",
        ])
        print_contact_api_status("/ur3/umi/finger1_1")
        print_contact_api_status("/ur3/umi/finger2_1")
        print_boundable_extents_under("/ur3/umi/finger1_1")
        print_boundable_extents_under("/ur3/umi/finger2_1")
        for prim_path, color in UNIT_TEST_CUBES:
            set_prim_color(prim_path, color)
            ensure_cube_ready(prim_path)
        print("cube ready")
        try:
            ensure_rigidbody_enabled("/ur3/Cube_UnitTest1", enable_gravity=True)
        except Exception as exc:
            print(f"[WARN] 启用 /ur3/Cube_UnitTest1 刚体属性失败: {exc}")
        # ensure_rigidbody_enabled("/ur3/Cube_UnitTest2", enable_gravity=False, kinematic=True)
        print("rigidbody enabled")
        cube1_sensors = attach_contact_sensors_selected_faces("/ur3/Cube_UnitTest1", "Cube1Contact", CONTACT_FACES_CUBE1)
        cube2_sensors = attach_contact_sensors_selected_faces("/ur3/Cube_UnitTest2", "Cube2Contact", CONTACT_FACES_CUBE2)
        cube1_count = sum(len(v) for v in cube1_sensors.values())
        cube2_count = sum(len(v) for v in cube2_sensors.values())
        print(f"[INFO] Cube1 传感器数量: {cube1_count}")
        print(f"[INFO] Cube2 传感器数量: {cube2_count}")
        cube1_posz_block_sensors = []
        if ENABLE_CUBE1_POSZ_BLOCKS:
            cube1_posz_block_sensors = attach_blocks_with_sensors_on_cube1_posz()
        if ENABLE_FINGER1_SENSORS:
            finger1_sensors = attach_contact_sensors_on_finger1()
            if VISUALIZE_SENSORS:
                _create_sensor_visuals({"finger1": finger1_sensors}, CONTACT_SENSOR_RADIUS)
        if VISUALIZE_SENSORS:
            _create_sensor_visuals(cube1_sensors, CONTACT_SENSOR_RADIUS)
            _create_sensor_visuals(cube2_sensors, CONTACT_SENSOR_RADIUS)
            if cube1_posz_block_sensors and VISUALIZE_BLOCK_SPHERES:
                _create_sensor_visuals({"cube1_posz_blocks": cube1_posz_block_sensors}, CONTACT_SENSOR_RADIUS)
        contact_sensor = (cube1_sensors.get("pos_y") or [None])[0]
        contact_sensor_cube2_neg_y = (cube2_sensors.get("neg_y") or [None])[0]
        # place_cube_above_other("/ur3/Cube_UnitTest1", "/ur3/Cube_UnitTest2")
        simulation_app.update()
        timeline = omni.timeline.get_timeline_interface()
        if AUTO_PLAY:
            timeline.play()
        initial_time = None
        print("before first update, is_running =", simulation_app.is_running())
        simulation_app.update()
        print("after first update,  is_running =", simulation_app.is_running())
        latest_contact_data = None
        pos_y_series = {}
        pos_y_contact_series = {}
        pos_y_force_history = {}
        pos_y_sensors = cube1_sensors.get("pos_y", [])
        last_plot_sample = 0.0
        plot_sampling_active = True
        plot_start_time = None
        force_plot_curve_map = {}
        last_force_plot_update = 0.0
        gripper_move_start_time = None
        finger1_start = None
        finger2_start = None
        gripper_move_done = False
        raycast_offsets = []
        raycast_cells = None
        last_raycast_update = 0.0
        finger1_raycast_offsets = []
        finger1_raycast_cells = None
        last_finger1_raycast_update = 0.0
        if ENABLE_RAYCAST_GRID:
            try:
                raycast_offsets = build_raycast_grid_offsets(
                    RAYCAST_PARENT, RAYCAST_FACE_LABEL, RAYCAST_GRID_SIZE, RAYCAST_PADDING
                )
                _ray_window, raycast_cells = _build_raycast_ui_window(
                    "Cube2 Smaller Raycast Distance", RAYCAST_GRID_SIZE, RAYCAST_GRID_SIZE
                )
                print(f"[INFO] 已在 {RAYCAST_PARENT} 创建 {RAYCAST_GRID_SIZE}x{RAYCAST_GRID_SIZE} Raycast 网格")
            except Exception as exc:
                print(f"[WARN] Raycast 网格初始化失败: {exc}")
        if ENABLE_FINGER1_RAYCAST_GRID:
            try:
                finger1_raycast_offsets = build_raycast_grid_offsets_rect(
                    FINGER1_RAYCAST_PARENT,
                    FINGER1_RAYCAST_FACE_LABEL,
                    FINGER1_RAYCAST_GRID[0],
                    FINGER1_RAYCAST_GRID[1],
                    FINGER1_RAYCAST_PADDING,
                )
                _finger1_ray_window, finger1_raycast_cells = _build_raycast_ui_window(
                    "Finger1 Raycast Distance", FINGER1_RAYCAST_GRID[0], FINGER1_RAYCAST_GRID[1]
                )
                print(
                    f"[INFO] 已在 {FINGER1_RAYCAST_PARENT} 创建 "
                    f"{FINGER1_RAYCAST_GRID[0]}x{FINGER1_RAYCAST_GRID[1]} Raycast 网格"
                )
                if VISUALIZE_FINGER1_RAYCAST_ORIGINS:
                    viz_paths = _create_raycast_origin_visuals(
                        FINGER1_RAYCAST_PARENT,
                        finger1_raycast_offsets,
                        FINGER1_RAYCAST_ORIGIN_SPHERE_RADIUS,
                        FINGER1_RAYCAST_ORIGIN_COLOR,
                        prefix="finger1_ray_origin",
                    )
                    if viz_paths:
                        print("[INFO] Finger1 raycast 可视化 prim:")
                        for path in viz_paths:
                            print(f"  {path}")
            except Exception as exc:
                print(f"[WARN] Finger1 Raycast 网格初始化失败: {exc}")
        if ENABLE_REALTIME_FORCE_PLOT and pos_y_sensors:
            _force_window, force_plot_curve_map = _build_force_plot_window(
                "Cube1 pos_y force (realtime)", GRID_SIZE, pos_y_sensors
            )
        _cube1_window, cube1_labels = _build_contact_ui_window(
            "Cube1 Contact (grid per face)", CONTACT_FACES_CUBE1
        )
        _cube2_window, cube2_labels = _build_contact_ui_window(
            "Cube2 Contact (grid per face)", CONTACT_FACES_CUBE2
        )
        block_labels = None
        if cube1_posz_block_sensors:
            _block_window, block_labels = _build_contact_ui_window(
                "Cube1 pos_z blocks", ["cube1_posz_blocks"], grid_size=CUBE1_POSZ_BLOCK_GRID[0]
            )
        last_ui_update = time.time()
        while simulation_app.is_running():
            simulation_app.update()
            if timeline.is_playing():
                if initial_time is None:
                    initial_time = time.time()
                if ENABLE_UMI_GRIPPER_OPEN and not gripper_move_done:
                    if gripper_move_start_time is None:
                        finger1_start = _get_local_translation("/ur3/umi/finger1_1")
                        finger2_start = _get_local_translation("/ur3/umi/finger2_1")
                        gripper_move_start_time = time.time()
                    duration = max(UMI_GRIPPER_OPEN_DURATION, 1e-6)
                    t = min((time.time() - gripper_move_start_time) / duration, 1.0)
                    new_finger1 = Gf.Vec3d(
                        float(finger1_start[0]) + UMI_FINGER1_X_DELTA * t,
                        float(finger1_start[1]),
                        float(finger1_start[2]),
                    )
                    new_finger2 = Gf.Vec3d(
                        float(finger2_start[0]) + UMI_FINGER2_X_DELTA * t,
                        float(finger2_start[1]),
                        float(finger2_start[2]),
                    )
                    _set_local_translation("/ur3/umi/finger1_1", new_finger1)
                    _set_local_translation("/ur3/umi/finger2_1", new_finger2)
                    if t >= 1.0:
                        gripper_move_done = True
                if PLOT_POS_Y_FORCE and plot_start_time is None:
                    plot_start_time = time.time()
                    last_plot_sample = plot_start_time
                latest_contact_data = log_contact_sensor_groups(
                    [("Cube1", cube1_sensors), ("Cube2", cube2_sensors)]
                )
                if PLOT_POS_Y_FORCE and pos_y_sensors:
                    now = time.time()
                    elapsed = now - plot_start_time if plot_start_time is not None else 0.0
                    if plot_sampling_active and elapsed > PLOT_MAX_DURATION:
                        plot_sampling_active = False
                        print(f"[INFO] pos_y 力-时间采样已达到上限 {PLOT_MAX_DURATION:.1f}s")
                    if plot_sampling_active and now - last_plot_sample >= PLOT_SAMPLE_INTERVAL:
                        _record_force_series(pos_y_sensors, pos_y_series, PLOT_MIN_FORCE)
                        _record_contact_series(pos_y_sensors, pos_y_contact_series)
                        last_plot_sample = now
                    _record_force_history(pos_y_sensors, pos_y_force_history, FORCE_PLOT_HISTORY)
                now = time.time()
                if now - last_ui_update >= UI_UPDATE_INTERVAL:
                    _update_contact_ui(cube1_labels, cube1_sensors)
                    _update_contact_ui(cube2_labels, cube2_sensors)
                    if block_labels:
                        _update_contact_ui(block_labels, {"cube1_posz_blocks": cube1_posz_block_sensors})
                    if VISUALIZE_SENSORS:
                        _update_sensor_visuals(cube1_sensors)
                        _update_sensor_visuals(cube2_sensors)
                        if cube1_posz_block_sensors and VISUALIZE_BLOCK_SPHERES:
                            _update_sensor_visuals({"cube1_posz_blocks": cube1_posz_block_sensors})
                    last_ui_update = now
                if (
                    ENABLE_REALTIME_FORCE_PLOT
                    and force_plot_curve_map
                    and (now - last_force_plot_update >= FORCE_PLOT_UPDATE_INTERVAL)
                ):
                    if FORCE_PLOT_AUTO_SCALE:
                        max_val = 0.0
                        for vals in pos_y_force_history.values():
                            if vals:
                                max_val = max(max_val, max(vals))
                        y_max = max(max_val, 1e-3)
                    else:
                        y_max = FORCE_PLOT_Y_MAX
                    _update_force_plot(force_plot_curve_map, pos_y_force_history, FORCE_PLOT_HISTORY, y_max)
                    last_force_plot_update = now
                if (
                    ENABLE_RAYCAST_GRID
                    and raycast_offsets
                    and raycast_cells
                    and (now - last_raycast_update >= RAYCAST_UI_UPDATE_INTERVAL)
                ):
                    distances = _query_raycast_distances(
                        RAYCAST_PARENT, raycast_offsets, RAYCAST_FACE_LABEL, RAYCAST_MAX_DISTANCE
                    )
                    _update_raycast_ui(raycast_cells, distances, RAYCAST_MAX_DISTANCE)
                    last_raycast_update = now
                if (
                    ENABLE_FINGER1_RAYCAST_GRID
                    and finger1_raycast_offsets
                    and finger1_raycast_cells
                    and (now - last_finger1_raycast_update >= FINGER1_RAYCAST_UI_UPDATE_INTERVAL)
                ):
                    distances = _query_raycast_distances(
                        FINGER1_RAYCAST_PARENT,
                        finger1_raycast_offsets,
                        FINGER1_RAYCAST_FACE_LABEL,
                        FINGER1_RAYCAST_MAX_DISTANCE,
                    )
                    _update_raycast_ui(finger1_raycast_cells, distances, FINGER1_RAYCAST_MAX_DISTANCE)
                    last_finger1_raycast_update = now
        if PLOT_POS_Y_FORCE:
            _plot_contact_force_combined(
                pos_y_sensors,
                pos_y_contact_series,
                pos_y_series,
                GRID_SIZE,
                PLOT_COMBINED_OUTPUT_PATH,
            )
        return latest_contact_data
    except Exception:
        print("[ERROR] main 中捕获到异常，堆栈如下：")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR] 捕获到异常，保持窗口运行以便观察。")
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.01)
