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


def _raycast_distances(
    physx_query,
    origins,
    direction,
    max_distance,
    ignore_rigid_body=None,
    collect_debug=False,
):
    distances = []
    debug_rows = []
    for r, c, origin in origins:
        hit = physx_query.raycast_closest(origin, direction, max_distance)
        hit_body = None
        hit_distance = None
        if hit and hit.get("hit"):
            hit_body = hit.get("rigidBody")
            hit_distance = float(hit.get("distance", 0.0))
            if ignore_rigid_body and hit_body == ignore_rigid_body:
                distances.append((r, c, None))
            else:
                distances.append((r, c, hit_distance))
        else:
            distances.append((r, c, None))
        if collect_debug:
            debug_rows.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "origin": tuple(float(v) for v in origin),
                    "direction": tuple(float(v) for v in direction),
                    "hit": bool(hit and hit.get("hit")),
                    "rigid_body": str(hit_body) if hit_body is not None else None,
                    "distance": hit_distance,
                    "ignored_self_hit": bool(ignore_rigid_body and hit_body == ignore_rigid_body),
                }
            )
    return distances, debug_rows


def _compute_contact_active(distances, max_distance):
    valid_distances = [float(d) for _, _, d in distances if d is not None]
    if not valid_distances:
        return False, None, 0
    min_distance = min(valid_distances)
    contact_active = min_distance < float(max_distance)
    return contact_active, min_distance, len(valid_distances)


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
    parser.add_argument("--ray-log-dir", type=str, default="test/ray_logs", help="Ray 日志输出目录")
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
        sphere_prim = _ensure_big_sphere(stage, cube_path=cube_path)
        _apply_rigidbody(sphere_prim, mass=3.0, enable_gravity=True, kinematic=False)
        _ensure_irregular_mesh(
            stage,
            seed=int(args.irregular_seed),
            size=float(args.irregular_size),
        )
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
    os.makedirs(args.ray_log_dir, exist_ok=True)
    ray_log_path = os.path.join(args.ray_log_dir, f"ray_distances_{run_id}.json")
    ray_log_csv = os.path.join(args.ray_log_dir, f"ray_distances_{run_id}.csv")
    force_log_csv = os.path.join(args.ray_log_dir, f"ray_forces_{run_id}.csv")
    delta_log_csv = os.path.join(args.ray_log_dir, f"ray_deltas_{run_id}.csv")
    delta_dot_log_csv = os.path.join(args.ray_log_dir, f"ray_delta_dots_{run_id}.csv")
    prim_pose_log_csv = os.path.join(args.ray_log_dir, f"prim_world_positions_{run_id}.csv")
    prim_transform_log_csv = os.path.join(args.ray_log_dir, f"prim_world_transforms_{run_id}.csv")
    contact_state_log_csv = os.path.join(args.ray_log_dir, f"contact_states_{run_id}.csv")
    tracked_prim_paths = ["/World/Cylinder_Test", "/World/BigRayCube"]

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
        contact_active, min_contact_distance, contact_ray_count = _compute_contact_active(
            distances,
            args.ray_max_distance,
        )
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
            "grid_size": args.ray_grid_size,
            "max_distance": args.ray_max_distance,
            "direction": args.ray_direction,
            "direction_world": world_ray_dir,
            "force_mapping": args.force_mapping,
            "spring_k": args.spring_k,
            "damping_c": args.damping_c,
            "distances": distances,
            "deltas": [(r, c, delta) for r, c, _, delta, _, _ in records],
            "delta_dots": [(r, c, delta_dot) for r, c, _, _, delta_dot, _ in records],
            "forces": [(r, c, f) for r, c, _, _, _, f in records],
            "prim_world_positions": {
                prim_path: (None if prim_world_transforms[prim_path] is None else prim_world_transforms[prim_path]["position"])
                for prim_path in tracked_prim_paths
            },
            "prim_world_transforms": prim_world_transforms,
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
                    args.ray_max_distance,
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
