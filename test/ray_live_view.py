import argparse
import json
import time
import os
import errno

import numpy as np
import matplotlib.pyplot as plt


def _load_delta_values(payload, distances, d_max):
    deltas = payload.get("deltas")
    if deltas:
        return deltas
    out = []
    for r, c, d in distances:
        if d is None:
            out.append((r, c, 0.0))
            continue
        d = float(d)
        if d > d_max:
            out.append((r, c, 0.0))
        else:
            out.append((r, c, max(0.0, d_max - d)))
    return out


def _load_delta_dot_values(payload, deltas):
    delta_dots = payload.get("delta_dots")
    if delta_dots:
        return delta_dots
    # Backward-compatible fallback for old logs: no delta_dot field.
    return [(r, c, 0.0) for r, c, _ in deltas]


def _load_force_values(payload, distances, d_max, force_max):
    forces = payload.get("forces")
    if forces:
        return forces
    # Backward-compatible fallback for old logs: reconstruct legacy force from distance.
    out = []
    for r, c, d in distances:
        if d is None:
            out.append((r, c, np.nan))
            continue
        d = float(d)
        if d > d_max:
            out.append((r, c, np.nan))
        else:
            force = max(0.0, (d_max - d) / max(d_max, 1e-6)) * force_max
            out.append((r, c, force))
    return out


def _parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="test/ray_logs/ray_distances.json", help="Ray 实时数据 JSON")
    parser.add_argument("--interval", type=float, default=0.1, help="刷新间隔(秒)")
    parser.add_argument("--vmin", type=float, default=0.0, help="颜色最小值(力)")
    parser.add_argument("--vmax", type=float, default=5.0, help="颜色最大值(力)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--show-text", action="store_true", help="在格子里显示数值")
    parser.add_argument("--show-distance", action="store_true", help="显示 distance 窗口（默认关闭）")
    parser.add_argument("--separate-windows", action="store_true", help="使用旧版分窗口显示（默认单窗口多子图）")
    parser.add_argument("--distance-max", type=float, default=None, help="距离阈值(超过则显示白)")
    parser.add_argument("--force-max", type=float, default=1.0, help="旧日志回退计算 force 时使用")
    parser.add_argument("--delta-vmax", type=float, default=None, help="delta 图颜色最大值")
    parser.add_argument("--delta-dot-vmin", type=float, default=-1.0, help="delta_dot 图颜色最小值")
    parser.add_argument("--delta-dot-vmax", type=float, default=1.0, help="delta_dot 图颜色最大值")
    parser.add_argument("--parent-pid", type=int, default=0, help="父进程 PID（父进程退出时自动关闭）")
    args = parser.parse_args()

    plt.ion()
    active_panels = ["force", "delta", "delta_dot"]
    if args.show_distance:
        active_panels = ["distance"] + active_panels
    panel_titles = {
        "distance": "distance",
        "force": "force",
        "delta": "delta",
        "delta_dot": "delta_dot",
    }
    panel_axes = {}
    panel_figs = {}
    if args.separate_windows:
        for key in active_panels:
            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(panel_titles[key])
            panel_axes[key] = ax
            panel_figs[key] = fig
    else:
        if args.show_distance:
            fig_main, axs = plt.subplots(2, 2)
            panel_axes = {
                "distance": axs[0, 0],
                "force": axs[0, 1],
                "delta": axs[1, 0],
                "delta_dot": axs[1, 1],
            }
        else:
            fig_main, axs = plt.subplots(1, 3)
            panel_axes = {
                "force": axs[0],
                "delta": axs[1],
                "delta_dot": axs[2],
            }
        fig_main.canvas.manager.set_window_title("ray_live_view")
        for key in active_panels:
            panel_figs[key] = fig_main

    panel_images = {k: None for k in active_panels}
    panel_text_grids = {k: [] for k in active_panels}

    def _panel_limits(name: str, d_max: float, delta_vmax: float):
        if name == "distance":
            return 0.0, d_max
        if name == "force":
            return args.vmin, args.vmax
        if name == "delta":
            return 0.0, delta_vmax
        return args.delta_dot_vmin, args.delta_dot_vmax

    try:
        while True:
            if not _parent_alive(args.parent_pid):
                break
            if not os.path.exists(args.json):
                time.sleep(args.interval)
                continue

            try:
                with open(args.json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                time.sleep(args.interval)
                continue

            grid_size = int(payload.get("grid_size", 10))
            max_distance = float(payload.get("max_distance", 0.5))
            d_max = max_distance if args.distance_max is None else float(args.distance_max)
            delta_vmax = d_max if args.delta_vmax is None else float(args.delta_vmax)
            distances = payload.get("distances", [])
            deltas = _load_delta_values(payload, distances, d_max)
            delta_dots = _load_delta_dot_values(payload, deltas)
            forces = _load_force_values(payload, distances, d_max, args.force_max)

            data_dist = np.full((grid_size, grid_size), np.nan, dtype=float)
            data_force = np.full((grid_size, grid_size), np.nan, dtype=float)
            data_delta = np.full((grid_size, grid_size), np.nan, dtype=float)
            data_delta_dot = np.full((grid_size, grid_size), np.nan, dtype=float)
            for r, c, d in distances:
                if d is None:
                    data_dist[int(r), int(c)] = np.nan
                else:
                    d = float(d)
                    if d > d_max:
                        data_dist[int(r), int(c)] = np.nan
                    else:
                        data_dist[int(r), int(c)] = d
            for r, c, delta in deltas:
                if delta is None:
                    data_delta[int(r), int(c)] = np.nan
                else:
                    data_delta[int(r), int(c)] = float(delta)
            for r, c, delta_dot in delta_dots:
                if delta_dot is None:
                    data_delta_dot[int(r), int(c)] = np.nan
                else:
                    data_delta_dot[int(r), int(c)] = float(delta_dot)
            for r, c, force in forces:
                if force is None:
                    data_force[int(r), int(c)] = np.nan
                else:
                    data_force[int(r), int(c)] = float(force)

            panel_data = {
                "distance": data_dist,
                "force": data_force,
                "delta": data_delta,
                "delta_dot": data_delta_dot,
            }
            for key in active_panels:
                ax = panel_axes[key]
                data = panel_data[key]
                vmin, vmax = _panel_limits(key, d_max, delta_vmax)
                if panel_images[key] is None:
                    cmap_obj = plt.get_cmap(args.cmap).copy()
                    cmap_obj.set_bad(color="white")
                    panel_images[key] = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap_obj)
                    ax.set_title(panel_titles[key])
                    ax.set_xlabel("col")
                    ax.set_ylabel("row")
                    panel_figs[key].colorbar(panel_images[key], ax=ax)
                    if args.show_text:
                        rows = []
                        for r in range(grid_size):
                            row = []
                            for c in range(grid_size):
                                t = ax.text(c, r, "", ha="center", va="center", color="white", fontsize=7)
                                row.append(t)
                            rows.append(row)
                        panel_text_grids[key] = rows
                else:
                    panel_images[key].set_data(data)
                panel_images[key].set_clim(vmin, vmax)

                if args.show_text and panel_text_grids[key]:
                    for r in range(grid_size):
                        for c in range(grid_size):
                            v = data[r, c]
                            panel_text_grids[key][r][c].set_text("" if np.isnan(v) else f"{v:.3f}")

            if args.separate_windows:
                for key in active_panels:
                    fig = panel_figs[key]
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            else:
                fig_main.canvas.draw()
                fig_main.canvas.flush_events()
            plt.pause(args.interval)
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
