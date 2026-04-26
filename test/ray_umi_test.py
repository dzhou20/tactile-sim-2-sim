"""Minimal Isaac Sim launcher for the UMI USD scene."""

import argparse
import atexit
import json
import math
import os
import signal
import subprocess
import sys
import time

from ray_utils import get_usd_path as _get_usd_path

# 不生成 __pycache__
sys.dont_write_bytecode = True

# Isaac Sim 4.5.0: use isaacsim.simulation_app (not omni.isaac.kit)
from isaacsim.simulation_app import SimulationApp

DEFAULT_UMI_USD_PATH = "assets/isaac_sim/create_asset/ur3_umi_0311.usd"
DEFAULT_TACTILE_SENSOR_FRAME_PATHS = (
    "/Robot/umi/finger1_1/tactile_sensor_frame",
    "/Robot/umi/finger2_1/tactile_sensor_frame",
)
DEFAULT_RAY_MARKER_OFFSET = 0.002
DEFAULT_RAY_MARKER_RADIUS = 0.0015
DEFAULT_RAY_GRID_ROWS = 5
DEFAULT_RAY_GRID_COLS = 5
DEFAULT_RAY_GRID_SPACING = 0.003
DEFAULT_RAY_GRID_POINT_RADIUS = 0.0008
DEFAULT_RAY_DIRECTION = "-x"
DEFAULT_RAY_MAX_DISTANCE = 0.001
DEFAULT_RAY_UPDATE_INTERVAL = 0.1
DEFAULT_DEBUG_RAY_INTERVAL = 1.0
DEFAULT_RAY_LOG_DIR = "data/ray_logs/umi"
DEFAULT_FORCE_MAX = 1.0
DEFAULT_FORCE_MAPPING = "spring_damper"
DEFAULT_SPRING_K = 10.0
DEFAULT_DAMPING_C = 5.0
DEFAULT_VIEW_VMIN = 0.0
DEFAULT_VIEW_VMAX = 5.0
DEFAULT_VIEW_CMAP = "inferno"
DEFAULT_VIEW_INTERVAL = 0.1


def _wait_for_stage_load(ctx, simulation_app: SimulationApp) -> None:
    if hasattr(ctx, "is_loading"):
        while ctx.is_loading():
            simulation_app.update()
    elif hasattr(ctx, "get_stage_state"):
        for _ in range(600):
            state = ctx.get_stage_state()
            if "OPEN" in str(state):
                break
            simulation_app.update()
    else:
        for _ in range(120):
            simulation_app.update()

    for _ in range(120):
        if ctx.get_stage() is not None:
            break
        simulation_app.update()


def _debug_print_stage_prims(stage) -> None:
    # Temporary debug helper: print the UMI stage prim tree to locate finger/body prim paths.
    # Keep this logic in ray_umi_test.py only; remove it after the target prim path is confirmed.
    print("[DEBUG] Stage prim traversal begin")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        type_name = prim.GetTypeName() or "<unknown>"
        depth = max(0, path.count("/") - 1)
        indent = "  " * depth
        print(f"[DEBUG] {indent}{path} ({type_name})")
    print("[DEBUG] Stage prim traversal end")


def _get_face_sign(face_selector: str) -> float:
    return 1.0 if str(face_selector).strip().lower() == "+x" else -1.0


def _get_sensor_body_path(sensor_frame_path: str) -> str:
    return str(sensor_frame_path).rsplit("/", 1)[0]


def _validate_tactile_sensor_frames(stage) -> None:
    missing_paths = []
    invalid_type_paths = []

    print("[INFO] Validating tactile sensor frame prims")
    for prim_path in DEFAULT_TACTILE_SENSOR_FRAME_PATHS:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            print(f"[ERROR] Missing tactile sensor frame: {prim_path}")
            missing_paths.append(prim_path)
            continue

        type_name = prim.GetTypeName() or "<unknown>"
        print(f"[INFO] Found tactile sensor frame: {prim_path} ({type_name})")
        if type_name != "Xform":
            invalid_type_paths.append((prim_path, type_name))

    if missing_paths or invalid_type_paths:
        error_lines = []
        if missing_paths:
            error_lines.append("Missing tactile sensor frame prims:")
            error_lines.extend(f"  - {path}" for path in missing_paths)
        if invalid_type_paths:
            error_lines.append("Tactile sensor frame prims must be Xform:")
            error_lines.extend(f"  - {path} (found: {type_name})" for path, type_name in invalid_type_paths)
        raise RuntimeError("\n".join(error_lines))


def _ensure_tactile_ray_marker(
    stage,
    sensor_frame_path: str,
    marker_name: str = "RayMarker",
    offset: float = DEFAULT_RAY_MARKER_OFFSET,
    radius: float = DEFAULT_RAY_MARKER_RADIUS,
    face_selector: str = DEFAULT_RAY_DIRECTION,
) -> None:
    from pxr import Gf, UsdGeom

    marker_path = f"{sensor_frame_path}/{marker_name}"
    marker = UsdGeom.Sphere.Define(stage, marker_path)
    face_sign = _get_face_sign(face_selector)
    signed_offset = face_sign * abs(float(offset))
    marker.GetRadiusAttr().Set(float(radius))
    marker.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.85, 0.2)])
    UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(signed_offset, 0.0, 0.0))
    print(
        f"[INFO] Ensured ray marker: {marker_path} "
        f"(local translate=({signed_offset:.6f}, 0.0, 0.0), direction={face_selector})"
    )


def _ensure_tactile_ray_markers(stage) -> None:
    for sensor_frame_path in DEFAULT_TACTILE_SENSOR_FRAME_PATHS:
        _ensure_tactile_ray_marker(stage, sensor_frame_path=sensor_frame_path)


def _ensure_tactile_ray_grid(
    stage,
    sensor_frame_path: str,
    grid_rows: int,
    grid_cols: int,
    grid_spacing: float,
    grid_offset: float,
    point_radius: float,
    face_selector: str,
    grid_parent_name: str = "RayGrid",
) -> None:
    # UMI-specific grid rebuild policy:
    # - Keep the RayGrid parent prim.
    # - Remove previously generated ray_*_* children before rebuilding.
    # - This differs from ray_open_test.py, which mainly redefines/reuses existing
    #   helper prims without an explicit cleanup pass.
    # Reason:
    # - Here we do not want stale tactile sample points to remain after changing
    #   rows/cols/spacing/offset/direction.
    #
    # UMI 专用的网格重建策略：
    # - 保留 RayGrid 父节点；
    # - 每次重建前先删除旧的 ray_*_* 子节点；
    # - 这与 ray_open_test.py 不同，后者主要是直接复用/重定义已有辅助 prim，
    #   没有显式做一次旧点清理。
    # 这样做的原因是：
    # - 这里不希望在修改 rows/cols/spacing/offset/direction 后留下旧的触觉采样点残留。
    from pxr import Gf, Sdf, UsdGeom

    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    spacing = float(grid_spacing)
    offset = _get_face_sign(face_selector) * abs(float(grid_offset))
    radius = float(point_radius)

    grid_parent_path = f"{sensor_frame_path}/{grid_parent_name}"
    grid_parent_prim = UsdGeom.Xform.Define(stage, grid_parent_path).GetPrim()

    # Remove previously generated ray_*_* children so the grid always reflects
    # the current rows/cols/spacing configuration without stale leftovers.
    for child_prim in list(grid_parent_prim.GetChildren()):
        child_name = child_prim.GetName()
        if child_name.startswith("ray_"):
            stage.RemovePrim(Sdf.Path(f"{grid_parent_path}/{child_name}"))

    y_start = -0.5 * spacing * (rows - 1)
    z_start = -0.5 * spacing * (cols - 1)

    for row in range(rows):
        for col in range(cols):
            y = y_start + spacing * row
            z = z_start + spacing * col
            ray_path = f"{grid_parent_path}/ray_{row}_{col}"
            sphere = UsdGeom.Sphere.Define(stage, ray_path)
            sphere.GetRadiusAttr().Set(radius)
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.95, 0.2)])
            UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(offset, y, z))

    print(
        f"[INFO] Ensured ray grid: {grid_parent_path} "
        f"(rows={rows}, cols={cols}, spacing={spacing:.6f}, offset={offset:.6f}, direction={face_selector})"
    )


def _ensure_tactile_ray_grids(
    stage,
    grid_rows: int,
    grid_cols: int,
    grid_spacing: float,
    grid_offset: float,
    point_radius: float,
    face_selector: str,
) -> None:
    for sensor_frame_path in DEFAULT_TACTILE_SENSOR_FRAME_PATHS:
        _ensure_tactile_ray_grid(
            stage,
            sensor_frame_path=sensor_frame_path,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_spacing=grid_spacing,
            grid_offset=grid_offset,
            point_radius=point_radius,
            face_selector=face_selector,
        )


def _build_tactile_grid_local_origins(
    grid_rows: int,
    grid_cols: int,
    grid_spacing: float,
    grid_offset: float,
    face_selector: str,
):
    rows = max(1, int(grid_rows))
    cols = max(1, int(grid_cols))
    spacing = float(grid_spacing)
    offset = _get_face_sign(face_selector) * abs(float(grid_offset))

    y_start = -0.5 * spacing * (rows - 1)
    z_start = -0.5 * spacing * (cols - 1)
    origins_local = []
    for row in range(rows):
        for col in range(cols):
            y = y_start + spacing * row
            z = z_start + spacing * col
            origins_local.append((row, col, (offset, y, z)))
    return origins_local


def _compute_raycast_frame(stage, sensor_frame_path, local_origins, face_selector):
    from pxr import Gf, Usd, UsdGeom

    sensor_frame_prim = stage.GetPrimAtPath(sensor_frame_path)
    if not sensor_frame_prim or not sensor_frame_prim.IsValid():
        return [], None

    xform = UsdGeom.Xformable(sensor_frame_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
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
    for row, col, local_origin in local_origins:
        local_pt = Gf.Vec3d(*local_origin)
        world_pt = xform.Transform(local_pt)
        world_origins.append((row, col, (float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))))
    return world_origins, world_dir


def _as_float3(value):
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return None


def _raycast_distances(
    physx_query,
    origins,
    direction,
    max_distance,
    ignore_rigid_body=None,
):
    distances = []
    ray_rows = []
    for row, col, origin in origins:
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
        if hit_detected and ignore_rigid_body and hit_body == ignore_rigid_body:
            distances.append((row, col, None))
        elif hit_detected:
            distances.append((row, col, hit_distance))
        else:
            distances.append((row, col, None))
        ray_rows.append(
            {
                "row": int(row),
                "col": int(col),
                "origin": tuple(float(v) for v in origin),
                "direction": tuple(float(v) for v in direction),
                "hit": hit_detected,
                "rigid_body": str(hit_body) if hit_body is not None else None,
                "distance": hit_distance,
                "position": hit_position,
                "normal": hit_normal,
                "ignored_self_hit": bool(ignore_rigid_body and hit_body == ignore_rigid_body),
            }
        )
    return distances, ray_rows


def _distance_to_delta(distance, max_distance):
    if distance is None:
        return 0.0
    if distance > max_distance:
        return 0.0
    return max(0.0, max_distance - distance)


def _delta_to_force_legacy(delta_value, max_distance, force_max):
    if delta_value is None:
        return 0.0
    return (delta_value / max(max_distance, 1e-6)) * force_max


def _delta_to_force_spring_damper(delta_value, delta_dot, spring_k, damping_c):
    if delta_value is None or delta_value <= 0.0:
        return 0.0
    fn = float(spring_k) * float(delta_value) + float(damping_c) * float(delta_dot)
    return max(0.0, fn)


class RaySignalProcessor:
    """Stateful processor for distance/delta/delta_dot/force signals."""

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
        if prev_delta is None or dt <= 1e-6:
            return 0.0
        return (curr_delta - prev_delta) / dt

    def update(self, distances, timestamp: float):
        now = float(timestamp)
        dt = 0.0 if self.prev_timestamp is None else max(0.0, now - self.prev_timestamp)
        records = []
        for row, col, distance in distances:
            delta_value = _distance_to_delta(distance, self.max_distance)
            prev_delta = self.prev_delta_map.get((int(row), int(col)))
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
            records.append((row, col, distance, delta_value, delta_dot, force_value))
        for row, col, _, delta_value, _, _ in records:
            self.prev_delta_map[(int(row), int(col))] = delta_value
        self.prev_timestamp = now
        return records, dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(keep_open=True)
    parser.add_argument("--keep-open", dest="keep_open", action="store_true", help="保持窗口打开直到手动关闭（默认）")
    parser.add_argument("--no-keep-open", dest="keep_open", action="store_false", help="按 close-after-frames 自动关闭")
    parser.add_argument("--close-after-frames", type=int, default=120, help="自动关闭前更新的帧数")
    parser.add_argument("--debug-print-prims", action="store_true", help="打印 stage prim 树（调试时临时使用）")
    parser.add_argument("--ray-marker-offset", type=float, default=DEFAULT_RAY_MARKER_OFFSET, help="RayMarker 沿局部 +X 的偏移")
    parser.add_argument("--ray-marker-radius", type=float, default=DEFAULT_RAY_MARKER_RADIUS, help="RayMarker 半径")
    parser.add_argument("--ray-grid-rows", type=int, default=DEFAULT_RAY_GRID_ROWS, help="RayGrid 行数")
    parser.add_argument("--ray-grid-cols", type=int, default=DEFAULT_RAY_GRID_COLS, help="RayGrid 列数")
    parser.add_argument("--ray-grid-spacing", type=float, default=DEFAULT_RAY_GRID_SPACING, help="RayGrid 相邻点间距")
    parser.add_argument("--ray-grid-offset", type=float, default=DEFAULT_RAY_MARKER_OFFSET, help="RayGrid 平面沿局部 +X 的偏移")
    parser.add_argument("--ray-grid-point-radius", type=float, default=DEFAULT_RAY_GRID_POINT_RADIUS, help="RayGrid 点半径")
    parser.add_argument("--ray-direction", type=str, default=DEFAULT_RAY_DIRECTION, choices=["+x", "-x"], help="Ray 方向")
    parser.add_argument("--ray-max-distance", type=float, default=DEFAULT_RAY_MAX_DISTANCE, help="Ray 最大接触距离")
    parser.add_argument("--ray-update-interval", type=float, default=DEFAULT_RAY_UPDATE_INTERVAL, help="Ray 数据刷新间隔(秒)")
    parser.add_argument("--debug-ray", action="store_true", help="打印 UMI Ray 调试信息")
    parser.add_argument("--debug-ray-interval", type=float, default=DEFAULT_DEBUG_RAY_INTERVAL, help="Ray 调试输出节流间隔(秒)")
    parser.add_argument("--ray-log-dir", type=str, default=DEFAULT_RAY_LOG_DIR, help="UMI Ray 日志输出目录")
    parser.add_argument("--force-max", type=float, default=DEFAULT_FORCE_MAX, help="旧线性映射(legacy)的最大力")
    parser.add_argument(
        "--force-mapping",
        type=str,
        default=DEFAULT_FORCE_MAPPING,
        choices=["spring_damper", "legacy"],
        help="法向力映射模式: spring_damper(默认) 或 legacy",
    )
    parser.add_argument("--spring-k", type=float, default=DEFAULT_SPRING_K, help="法向弹簧刚度 k (N/m)")
    parser.add_argument("--damping-c", type=float, default=DEFAULT_DAMPING_C, help="法向阻尼系数 c (N·s/m)")
    parser.add_argument("--auto-view", action="store_true", help="自动启动外部可视化")
    parser.add_argument("--no-auto-view", action="store_true", help="不自动启动外部可视化")
    parser.add_argument("--view-vmin", type=float, default=DEFAULT_VIEW_VMIN, help="可视化颜色最小值")
    parser.add_argument("--view-vmax", type=float, default=DEFAULT_VIEW_VMAX, help="可视化颜色最大值")
    parser.add_argument("--view-cmap", type=str, default=DEFAULT_VIEW_CMAP, help="可视化 colormap")
    parser.add_argument("--view-interval", type=float, default=DEFAULT_VIEW_INTERVAL, help="可视化刷新间隔(秒)")
    parser.add_argument(
        "--usd-path",
        type=str,
        default=DEFAULT_UMI_USD_PATH,
        help="UMI USD 场景路径；默认打开 ur3_umi_0311.usd",
    )
    args = parser.parse_args()

    simulation_app = SimulationApp({"headless": False})

    import omni.usd
    from pxr import UsdGeom

    usd_path = _get_usd_path(args.usd_path)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    ctx = omni.usd.get_context()
    opened = ctx.open_stage(usd_path)
    if not opened:
        raise RuntimeError(f"无法打开 USD: {usd_path}")

    _wait_for_stage_load(ctx, simulation_app)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Stage 加载失败: {usd_path}")

    default_prim = stage.GetDefaultPrim()
    default_prim_path = default_prim.GetPath().pathString if default_prim and default_prim.IsValid() else "<none>"
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    print(f"[INFO] Loaded UMI USD: {usd_path}")
    print(f"[INFO] Stage default prim: {default_prim_path}")
    print(f"[INFO] Stage metersPerUnit: {meters_per_unit}")
    _validate_tactile_sensor_frames(stage)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    ray_log_dir = os.path.abspath(args.ray_log_dir)
    os.makedirs(ray_log_dir, exist_ok=True)
    write_combined_ray_log = False
    combined_ray_log_path = (
        os.path.join(ray_log_dir, f"umi_combined_{run_id}.json")
        if write_combined_ray_log
        else None
    )
    sensor_configs = []
    for sensor_idx, sensor_frame_path in enumerate(DEFAULT_TACTILE_SENSOR_FRAME_PATHS):
        _ensure_tactile_ray_marker(
            stage,
            sensor_frame_path=sensor_frame_path,
            offset=float(args.ray_marker_offset),
            radius=float(args.ray_marker_radius),
            face_selector=str(args.ray_direction),
        )
        sensor_configs.append(
            {
                "sensor_index": sensor_idx,
                "sensor_frame_path": sensor_frame_path,
                "sensor_body_path": _get_sensor_body_path(sensor_frame_path),
                "sensor_name": os.path.basename(_get_sensor_body_path(sensor_frame_path)),
                "local_origins": _build_tactile_grid_local_origins(
                    grid_rows=int(args.ray_grid_rows),
                    grid_cols=int(args.ray_grid_cols),
                    grid_spacing=float(args.ray_grid_spacing),
                    grid_offset=float(args.ray_grid_offset),
                    face_selector=str(args.ray_direction),
                ),
                "signal_processor": RaySignalProcessor(
                    max_distance=float(args.ray_max_distance),
                    force_max=float(args.force_max),
                    force_mapping=str(args.force_mapping),
                    spring_k=float(args.spring_k),
                    damping_c=float(args.damping_c),
                ),
                "ray_log_path": os.path.join(
                    ray_log_dir,
                    f"umi_{os.path.basename(_get_sensor_body_path(sensor_frame_path))}_{run_id}.json",
                ),
            }
        )
    _ensure_tactile_ray_grids(
        stage,
        grid_rows=int(args.ray_grid_rows),
        grid_cols=int(args.ray_grid_cols),
        grid_spacing=float(args.ray_grid_spacing),
        grid_offset=float(args.ray_grid_offset),
        point_radius=float(args.ray_grid_point_radius),
        face_selector=str(args.ray_direction),
    )
    print(
        f"[INFO] UMI ray sensors active: count={len(sensor_configs)} "
        f"direction={args.ray_direction} max_distance={float(args.ray_max_distance):.6f}"
    )
    if args.debug_print_prims:
        _debug_print_stage_prims(stage)

    auto_view = args.auto_view or not args.no_auto_view
    view_processes = []
    if auto_view:
        view_script = os.path.join(os.path.dirname(__file__), "ray_live_view.py")
        for sensor in sensor_configs:
            cmd = [
                sys.executable,
                view_script,
                "--json",
                sensor["ray_log_path"],
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
                "--hide-tangential",
            ]
            try:
                view_processes.append(subprocess.Popen(cmd))
            except Exception:
                pass

    def _cleanup_views():
        for proc in view_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    atexit.register(_cleanup_views)

    import omni.physx

    physx_query = omni.physx.get_physx_scene_query_interface()
    last_ray_update = 0.0
    last_debug_print = 0.0
    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def _update_ray_sensors(now: float) -> None:
        nonlocal last_ray_update
        nonlocal last_debug_print

        summaries = []
        if write_combined_ray_log:
            combined_distances = []
            combined_deltas = []
            combined_delta_dots = []
            combined_forces = []
            combined_dt = 0.0
        for sensor in sensor_configs:
            world_origins, world_dir = _compute_raycast_frame(
                stage,
                sensor["sensor_frame_path"],
                sensor["local_origins"],
                args.ray_direction,
            )
            if not world_origins or world_dir is None:
                summaries.append(
                    {
                        "sensor_frame_path": sensor["sensor_frame_path"],
                        "valid": False,
                    }
                )
                continue

            distances, ray_rows = _raycast_distances(
                physx_query,
                world_origins,
                world_dir,
                float(args.ray_max_distance),
                ignore_rigid_body=sensor["sensor_body_path"],
            )
            records, dt = sensor["signal_processor"].update(distances, now)
            if write_combined_ray_log:
                combined_dt = dt
            non_none_distances = [d for _, _, d in distances if d is not None]
            hit_count = sum(1 for row in ray_rows if row["hit"])
            payload = {
                "timestamp": now,
                "delta_t": dt,
                "sensor_frame_path": sensor["sensor_frame_path"],
                "sensor_body_path": sensor["sensor_body_path"],
                "grid_size": max(int(args.ray_grid_rows), int(args.ray_grid_cols)),
                "grid_rows": int(args.ray_grid_rows),
                "grid_cols": int(args.ray_grid_cols),
                "max_distance": float(args.ray_max_distance),
                "direction": args.ray_direction,
                "direction_world": world_dir,
                "force_mapping": args.force_mapping,
                "spring_k": float(args.spring_k),
                "damping_c": float(args.damping_c),
                "distances": distances,
                "deltas": [(row, col, delta) for row, col, _, delta, _, _ in records],
                "delta_dots": [(row, col, delta_dot) for row, col, _, _, delta_dot, _ in records],
                "forces": [(row, col, force) for row, col, _, _, _, force in records],
                "taxel_d_t_in_A": [],
                "taxel_contact_states": [],
            }
            with open(sensor["ray_log_path"], "w", encoding="utf-8") as f:
                json.dump(payload, f)
            if write_combined_ray_log:
                col_offset = int(sensor["sensor_index"]) * int(args.ray_grid_cols)
                combined_distances.extend(
                    [(row, col + col_offset, distance) for row, col, distance in distances]
                )
                combined_deltas.extend(
                    [(row, col + col_offset, delta) for row, col, _, delta, _, _ in records]
                )
                combined_delta_dots.extend(
                    [(row, col + col_offset, delta_dot) for row, col, _, _, delta_dot, _ in records]
                )
                combined_forces.extend(
                    [(row, col + col_offset, force) for row, col, _, _, _, force in records]
                )
            summaries.append(
                {
                    "sensor_frame_path": sensor["sensor_frame_path"],
                    "sensor_body_path": sensor["sensor_body_path"],
                    "ray_log_path": sensor["ray_log_path"],
                    "valid": True,
                    "world_dir": world_dir,
                    "ray_count": len(distances),
                    "hit_count": hit_count,
                    "valid_hit_count": len(non_none_distances),
                    "min_distance": None if not non_none_distances else min(non_none_distances),
                    "max_distance": None if not non_none_distances else max(non_none_distances),
                    "max_force": max((force for _, _, _, _, _, force in records), default=0.0),
                    "max_delta": max((delta for _, _, _, delta, _, _ in records), default=0.0),
                    "max_delta_dot": max((abs(delta_dot) for _, _, _, _, delta_dot, _ in records), default=0.0),
                }
            )

        if write_combined_ray_log:
            combined_payload = {
                "timestamp": now,
                "delta_t": combined_dt,
                "grid_size": int(args.ray_grid_cols) * len(sensor_configs),
                "grid_rows": int(args.ray_grid_rows),
                "grid_cols": int(args.ray_grid_cols) * len(sensor_configs),
                "max_distance": float(args.ray_max_distance),
                "direction": args.ray_direction,
                "force_mapping": args.force_mapping,
                "spring_k": float(args.spring_k),
                "damping_c": float(args.damping_c),
                "distances": combined_distances,
                "deltas": combined_deltas,
                "delta_dots": combined_delta_dots,
                "forces": combined_forces,
                "taxel_d_t_in_A": [],
                "taxel_contact_states": [],
                "sensor_names": [sensor["sensor_name"] for sensor in sensor_configs],
                "sensor_column_slices": [
                    {
                        "sensor_name": sensor["sensor_name"],
                        "col_start": int(sensor["sensor_index"]) * int(args.ray_grid_cols),
                        "col_end": (int(sensor["sensor_index"]) + 1) * int(args.ray_grid_cols) - 1,
                    }
                    for sensor in sensor_configs
                ],
            }
            with open(combined_ray_log_path, "w", encoding="utf-8") as f:
                json.dump(combined_payload, f)

        if args.debug_ray and (now - last_debug_print >= max(0.0, float(args.debug_ray_interval))):
            for summary in summaries:
                if not summary["valid"]:
                    print(f"[DEBUG] UMI ray sensor invalid: {summary['sensor_frame_path']}")
                    continue
                min_distance = summary["min_distance"]
                max_distance = summary["max_distance"]
                min_text = "none" if min_distance is None else f"{float(min_distance):.6f}"
                max_text = "none" if max_distance is None else f"{float(max_distance):.6f}"
                print(
                    "[DEBUG] UMI ray summary: "
                    f"frame={summary['sensor_frame_path']} "
                    f"body={summary['sensor_body_path']} "
                    f"dir={tuple(round(v, 6) for v in summary['world_dir'])} "
                    f"rays={summary['ray_count']} "
                    f"hits={summary['hit_count']} "
                    f"valid_hits={summary['valid_hit_count']} "
                    f"min={min_text} "
                    f"max={max_text} "
                    f"max_delta={float(summary['max_delta']):.6f} "
                    f"max_delta_dot={float(summary['max_delta_dot']):.6f} "
                    f"max_force={float(summary['max_force']):.6f}"
                )
            last_debug_print = now
        last_ray_update = now

    try:
        if args.keep_open:
            while (not stop_requested) and simulation_app.is_running():
                simulation_app.update()
                now = time.time()
                if now - last_ray_update >= max(0.0, float(args.ray_update_interval)):
                    _update_ray_sensors(now)
        else:
            for _ in range(max(1, int(args.close_after_frames))):
                if stop_requested:
                    break
                simulation_app.update()
                now = time.time()
                if now - last_ray_update >= max(0.0, float(args.ray_update_interval)):
                    _update_ray_sensors(now)
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
