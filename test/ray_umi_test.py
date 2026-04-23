"""Minimal Isaac Sim launcher for the UMI USD scene."""

import argparse
import os
import signal
import sys

from ray_utils import get_usd_path as _get_usd_path

# 不生成 __pycache__
sys.dont_write_bytecode = True

# Isaac Sim 4.5.0: use isaacsim.simulation_app (not omni.isaac.kit)
from isaacsim.simulation_app import SimulationApp

DEFAULT_UMI_USD_PATH = "assets/isaac_sim/create_asset/ur3_umi_0311.usd"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(keep_open=True)
    parser.add_argument("--keep-open", dest="keep_open", action="store_true", help="保持窗口打开直到手动关闭（默认）")
    parser.add_argument("--no-keep-open", dest="keep_open", action="store_false", help="按 close-after-frames 自动关闭")
    parser.add_argument("--close-after-frames", type=int, default=120, help="自动关闭前更新的帧数")
    parser.add_argument("--debug-print-prims", action="store_true", help="打印 stage prim 树（调试时临时使用）")
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
    if args.debug_print_prims:
        _debug_print_stage_prims(stage)

    stop_requested = False

    def _request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        if args.keep_open:
            while (not stop_requested) and simulation_app.is_running():
                simulation_app.update()
        else:
            for _ in range(max(1, int(args.close_after_frames))):
                if stop_requested:
                    break
                simulation_app.update()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
