import argparse
import json
import time
import os
import errno

import numpy as np
import matplotlib.pyplot as plt


def _load_insert_values(payload, distances, d_max):
    inserts = payload.get("inserts")
    if inserts:
        return inserts
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
    parser.add_argument("--vmax", type=float, default=1.0, help="颜色最大值(力)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--show-text", action="store_true", help="在格子里显示数值")
    parser.add_argument("--distance-max", type=float, default=None, help="距离阈值(超过则显示白)")
    parser.add_argument("--force-max", type=float, default=1.0, help="距离->力 映射的最大力")
    parser.add_argument("--insert-vmax", type=float, default=None, help="insert 图颜色最大值")
    parser.add_argument("--parent-pid", type=int, default=0, help="父进程 PID（父进程退出时自动关闭）")
    args = parser.parse_args()

    plt.ion()
    fig_dist, ax_dist = plt.subplots()
    fig_force, ax_force = plt.subplots()
    fig_insert, ax_insert = plt.subplots()
    fig_dist.canvas.manager.set_window_title("distance")
    fig_force.canvas.manager.set_window_title("force")
    fig_insert.canvas.manager.set_window_title("insert")
    img_dist = None
    img_force = None
    img_insert = None
    text_grid_dist = []
    text_grid_force = []
    text_grid_insert = []

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
            insert_vmax = d_max if args.insert_vmax is None else float(args.insert_vmax)
            distances = payload.get("distances", [])
            inserts = _load_insert_values(payload, distances, d_max)

            data_dist = np.full((grid_size, grid_size), np.nan, dtype=float)
            data_force = np.full((grid_size, grid_size), np.nan, dtype=float)
            data_insert = np.full((grid_size, grid_size), np.nan, dtype=float)
            for r, c, d in distances:
                if d is None:
                    data_dist[int(r), int(c)] = np.nan
                    data_force[int(r), int(c)] = np.nan
                else:
                    d = float(d)
                    if d > d_max:
                        data_dist[int(r), int(c)] = np.nan
                        data_force[int(r), int(c)] = np.nan
                    else:
                        # 与 ray_open_test.py 保持一致的线性映射:
                        # insert = max(0, d_max - d)
                        # force  = (insert / d_max) * force_max
                        #       = max(0, (d_max - d) / d_max) * force_max
                        force = max(0.0, (d_max - d) / max(d_max, 1e-6)) * args.force_max
                        data_dist[int(r), int(c)] = d
                        data_force[int(r), int(c)] = force
            for r, c, ins in inserts:
                if ins is None:
                    data_insert[int(r), int(c)] = np.nan
                else:
                    data_insert[int(r), int(c)] = float(ins)

            if img_dist is None:
                cmap_dist = plt.get_cmap(args.cmap).copy()
                cmap_dist.set_bad(color="white")
                img_dist = ax_dist.imshow(data_dist, vmin=0.0, vmax=d_max, cmap=cmap_dist)
                ax_dist.set_title("distance")
                ax_dist.set_xlabel("col")
                ax_dist.set_ylabel("row")
                fig_dist.colorbar(img_dist, ax=ax_dist)
                if args.show_text:
                    for r in range(grid_size):
                        row = []
                        for c in range(grid_size):
                            t = ax_dist.text(c, r, "", ha="center", va="center", color="white", fontsize=7)
                            row.append(t)
                        text_grid_dist.append(row)
            else:
                img_dist.set_data(data_dist)

            if img_force is None:
                cmap_force = plt.get_cmap(args.cmap).copy()
                cmap_force.set_bad(color="white")
                img_force = ax_force.imshow(data_force, vmin=args.vmin, vmax=args.vmax, cmap=cmap_force)
                ax_force.set_title("force")
                ax_force.set_xlabel("col")
                ax_force.set_ylabel("row")
                fig_force.colorbar(img_force, ax=ax_force)
                if args.show_text:
                    for r in range(grid_size):
                        row = []
                        for c in range(grid_size):
                            t = ax_force.text(c, r, "", ha="center", va="center", color="white", fontsize=7)
                            row.append(t)
                        text_grid_force.append(row)
            else:
                img_force.set_data(data_force)

            if img_insert is None:
                cmap_insert = plt.get_cmap(args.cmap).copy()
                cmap_insert.set_bad(color="white")
                img_insert = ax_insert.imshow(data_insert, vmin=0.0, vmax=insert_vmax, cmap=cmap_insert)
                ax_insert.set_title("insert")
                ax_insert.set_xlabel("col")
                ax_insert.set_ylabel("row")
                fig_insert.colorbar(img_insert, ax=ax_insert)
                if args.show_text:
                    for r in range(grid_size):
                        row = []
                        for c in range(grid_size):
                            t = ax_insert.text(c, r, "", ha="center", va="center", color="white", fontsize=7)
                            row.append(t)
                        text_grid_insert.append(row)
            else:
                img_insert.set_data(data_insert)

            if args.show_text and text_grid_dist:
                for r in range(grid_size):
                    for c in range(grid_size):
                        v = data_dist[r, c]
                        text_grid_dist[r][c].set_text("" if np.isnan(v) else f"{v:.3f}")
            if args.show_text and text_grid_force:
                for r in range(grid_size):
                    for c in range(grid_size):
                        v = data_force[r, c]
                        text_grid_force[r][c].set_text("" if np.isnan(v) else f"{v:.3f}")
            if args.show_text and text_grid_insert:
                for r in range(grid_size):
                    for c in range(grid_size):
                        v = data_insert[r, c]
                        text_grid_insert[r][c].set_text("" if np.isnan(v) else f"{v:.3f}")

            img_dist.set_clim(0.0, d_max)
            img_force.set_clim(args.vmin, args.vmax)
            img_insert.set_clim(0.0, insert_vmax)
            fig_dist.canvas.draw()
            fig_force.canvas.draw()
            fig_insert.canvas.draw()
            fig_dist.canvas.flush_events()
            fig_force.canvas.flush_events()
            fig_insert.canvas.flush_events()
            plt.pause(args.interval)
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
