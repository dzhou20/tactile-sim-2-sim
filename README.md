# tactile-sim-2-sim

## Ray 可视化说明（ray_open_test.py + ray_live_view.py）
- 正方体：`/World/BigRayCube`
- 记录小球（正 X 方向）：`/World/RayMarker`
- Ray 网格点父节点：`/World/RayGrid`

方向与索引约定：
- Ray 起点位于正方体 **+X 面**外侧 `ray-padding` 米处（默认 0.002m）。
- Ray 方向默认 **-X**（可用 `--ray-direction +x` 改为 +X）。
- 10×10 网格铺在 **Y-Z 平面**：`row` 沿 +Y 增长，`col` 沿 +Z 增长。

外部可视化（非 Isaac Sim 原生 UI）：
```
python test/ray_open_test.py --keep-open --ray-padding 0.0005
python test/ray_live_view.py --vmin 0 --vmax 0.5 --cmap inferno --interval 0.1
```
