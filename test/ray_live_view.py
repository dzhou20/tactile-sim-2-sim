import argparse
import json
import time
import os
import errno

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def _get_repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_get_repo_root(), path)


def _load_tangential_values(payload):
    tangential = payload.get("taxel_d_t_in_A")
    if tangential:
        return tangential
    return []


def _load_contact_states(payload):
    states = payload.get("taxel_contact_states")
    if states:
        return states
    return []


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
    parser.add_argument("--json", type=str, default="data/ray_logs/ray_distances.json", help="Ray 实时数据 JSON")
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
    parser.add_argument("--tangential-vmax", type=float, default=None, help="|d_t_A| 图颜色最大值")
    parser.add_argument("--quiver-scale", type=float, default=1000.0, help="切向箭头缩放因子")
    parser.add_argument("--hide-tangential", action="store_true", help="隐藏切向面板（仅显示法向相关）")
    parser.add_argument("--parent-pid", type=int, default=0, help="父进程 PID（父进程退出时自动关闭）")
    args = parser.parse_args()
    args.json = _resolve_repo_path(args.json)

    plt.ion()
    if args.hide_tangential:
        active_panels = ["force", "delta", "delta_dot", "contact_state"]
    else:
        active_panels = ["force", "delta", "delta_dot", "tangential_mag", "tangential_vec", "contact_state"]
    if args.show_distance:
        active_panels = ["distance"] + active_panels
    panel_titles = {
        "distance": "distance",
        "force": "force",
        "delta": "delta",
        "delta_dot": "delta_dot",
        "tangential_mag": "|d_t_A|",
        "tangential_vec": "(d_t_A_y, d_t_A_z)",
        "contact_state": "contact_state",
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
        if args.hide_tangential:
            if args.show_distance:
                fig_main, axs = plt.subplots(1, 5)
                flat_axes = axs.flatten()
                panel_axes = {
                    "distance": flat_axes[0],
                    "force": flat_axes[1],
                    "delta": flat_axes[2],
                    "delta_dot": flat_axes[3],
                    "contact_state": flat_axes[4],
                }
            else:
                fig_main, axs = plt.subplots(2, 2)
                flat_axes = axs.flatten()
                panel_axes = {
                    "force": flat_axes[0],
                    "delta": flat_axes[1],
                    "delta_dot": flat_axes[2],
                    "contact_state": flat_axes[3],
                }
        else:
            if args.show_distance:
                fig_main, axs = plt.subplots(3, 3)
                flat_axes = axs.flatten()
                panel_axes = {
                    "distance": flat_axes[0],
                    "force": flat_axes[1],
                    "delta": flat_axes[2],
                    "delta_dot": flat_axes[3],
                    "tangential_mag": flat_axes[4],
                    "tangential_vec": flat_axes[5],
                    "contact_state": flat_axes[6],
                }
                flat_axes[7].axis("off")
                flat_axes[8].axis("off")
            else:
                fig_main, axs = plt.subplots(2, 3)
                flat_axes = axs.flatten()
                panel_axes = {
                    "force": flat_axes[0],
                    "delta": flat_axes[1],
                    "delta_dot": flat_axes[2],
                    "tangential_mag": flat_axes[3],
                    "tangential_vec": flat_axes[4],
                    "contact_state": flat_axes[5],
                }
        fig_main.canvas.manager.set_window_title("ray_live_view")
        for key in active_panels:
            panel_figs[key] = fig_main

    panel_images = {k: None for k in active_panels}
    panel_text_grids = {k: [] for k in active_panels}
    panel_quivers = {k: None for k in active_panels}
    contact_state_map = {"none": 0.0, "start": 1.0, "hold": 2.0, "end": 3.0}
    contact_state_labels = {0.0: "none", 1.0: "start", 2.0: "hold", 3.0: "end"}
    contact_state_cmap = mcolors.ListedColormap(["white", "#2a9d8f", "#e9c46a", "#e76f51"])
    contact_state_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], contact_state_cmap.N)

    def _panel_limits(name: str, d_max: float, delta_vmax: float, tangential_vmax: float):
        if name == "distance":
            return 0.0, d_max
        if name == "force":
            return args.vmin, args.vmax
        if name == "delta":
            return 0.0, delta_vmax
        if name == "tangential_mag":
            return 0.0, tangential_vmax
        if name == "contact_state":
            return -0.5, 3.5
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

            grid_rows = int(payload.get("grid_rows", payload.get("grid_size", 10)))
            grid_cols = int(payload.get("grid_cols", payload.get("grid_size", 10)))
            max_distance = float(payload.get("max_distance", 0.5))
            d_max = max_distance if args.distance_max is None else float(args.distance_max)
            delta_vmax = d_max if args.delta_vmax is None else float(args.delta_vmax)
            tangential_vmax = d_max if args.tangential_vmax is None else float(args.tangential_vmax)
            distances = payload.get("distances", [])
            deltas = _load_delta_values(payload, distances, d_max)
            delta_dots = _load_delta_dot_values(payload, deltas)
            forces = _load_force_values(payload, distances, d_max, args.force_max)
            tangential = _load_tangential_values(payload)
            contact_states = _load_contact_states(payload)

            data_dist = np.full((grid_rows, grid_cols), np.nan, dtype=float)
            data_force = np.full((grid_rows, grid_cols), np.nan, dtype=float)
            data_delta = np.full((grid_rows, grid_cols), np.nan, dtype=float)
            data_delta_dot = np.full((grid_rows, grid_cols), np.nan, dtype=float)
            data_tangential_mag = np.full((grid_rows, grid_cols), np.nan, dtype=float)
            data_tangential_y = np.zeros((grid_rows, grid_cols), dtype=float)
            data_tangential_z = np.zeros((grid_rows, grid_cols), dtype=float)
            data_contact_state = np.zeros((grid_rows, grid_cols), dtype=float)
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
            for r, c, d_t in tangential:
                rr = int(r)
                cc = int(c)
                if d_t is None:
                    data_tangential_mag[rr, cc] = np.nan
                    data_tangential_y[rr, cc] = 0.0
                    data_tangential_z[rr, cc] = 0.0
                    continue
                dy = float(d_t[1])
                dz = float(d_t[2])
                data_tangential_y[rr, cc] = dy
                data_tangential_z[rr, cc] = dz
                data_tangential_mag[rr, cc] = float(np.sqrt(dy * dy + dz * dz))
            for state in contact_states:
                rr = int(state["row"])
                cc = int(state["col"])
                data_contact_state[rr, cc] = contact_state_map.get(str(state.get("last_event", "none")), 0.0)

            panel_data = {
                "distance": data_dist,
                "force": data_force,
                "delta": data_delta,
                "delta_dot": data_delta_dot,
                "tangential_mag": data_tangential_mag,
                "contact_state": data_contact_state,
            }
            for key in active_panels:
                ax = panel_axes[key]
                if key == "tangential_vec":
                    u = data_tangential_z * float(args.quiver_scale)
                    v = data_tangential_y * float(args.quiver_scale)
                    valid_mask = ~np.isnan(data_tangential_mag)
                    if panel_quivers[key] is None:
                        xs, ys = np.meshgrid(np.arange(grid_cols), np.arange(grid_rows))
                        ax.set_title(panel_titles[key])
                        ax.set_xlabel("col")
                        ax.set_ylabel("row")
                        ax.set_xlim(-0.5, grid_cols - 0.5)
                        ax.set_ylim(grid_rows - 0.5, -0.5)
                        ax.set_aspect("equal")
                        ax.set_facecolor("white")
                        panel_quivers[key] = ax.quiver(
                            xs,
                            ys,
                            np.where(valid_mask, u, 0.0),
                            np.where(valid_mask, v, 0.0),
                            angles="xy",
                            scale_units="xy",
                            scale=1.0,
                            color="black",
                        )
                    else:
                        panel_quivers[key].set_UVC(
                            np.where(valid_mask, u, 0.0),
                            np.where(valid_mask, v, 0.0),
                        )
                    continue
                data = panel_data[key]
                vmin, vmax = _panel_limits(key, d_max, delta_vmax, tangential_vmax)
                if panel_images[key] is None:
                    if key == "contact_state":
                        panel_images[key] = ax.imshow(data, cmap=contact_state_cmap, norm=contact_state_norm)
                    else:
                        cmap_obj = plt.get_cmap(args.cmap).copy()
                        cmap_obj.set_bad(color="white")
                        panel_images[key] = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap_obj)
                    ax.set_title(panel_titles[key])
                    ax.set_xlabel("col")
                    ax.set_ylabel("row")
                    cbar = panel_figs[key].colorbar(panel_images[key], ax=ax)
                    if key == "contact_state":
                        cbar.set_ticks([0.0, 1.0, 2.0, 3.0])
                        cbar.set_ticklabels(["none", "start", "hold", "end"])
                    if args.show_text:
                        rows = []
                        for r in range(grid_rows):
                            row = []
                            for c in range(grid_cols):
                                t = ax.text(c, r, "", ha="center", va="center", color="white", fontsize=7)
                                row.append(t)
                            rows.append(row)
                        panel_text_grids[key] = rows
                else:
                    panel_images[key].set_data(data)
                if key != "contact_state":
                    panel_images[key].set_clim(vmin, vmax)

                if args.show_text and panel_text_grids[key]:
                    for r in range(grid_rows):
                        for c in range(grid_cols):
                            v = data[r, c]
                            if key == "contact_state":
                                panel_text_grids[key][r][c].set_text(contact_state_labels.get(v, ""))
                            else:
                                panel_text_grids[key][r][c].set_text("" if np.isnan(v) else f"{v:.3f}")

            if args.separate_windows:
                for key in active_panels:
                    fig = panel_figs[key]
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
            else:
                fig_main.canvas.draw_idle()
                fig_main.canvas.flush_events()
            time.sleep(args.interval)
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
