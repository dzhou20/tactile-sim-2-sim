# tactile-sim-2-sim

## Quick Start | 快速开始

Run from repo root:

```bash
python test/ray_open_test.py --keep-open --ray-padding 0.0005
```

This starts Isaac Sim ray sampling and writes logs (`distance`, `delta`, `delta_dot`, `force`) to `test/ray_logs/`.
该命令启动 Isaac Sim 射线采样，并在 `test/ray_logs/` 写出日志（`distance`、`delta`、`delta_dot`、`force`）。

## Commands | 命令说明

### 1) Main simulation script | 主仿真脚本

```bash
python test/ray_open_test.py [options]
```

Important options | 常用参数:
- `--keep-open`: keep running until manual close. | 保持运行直到手动关闭。
- `--ray-padding <meters>`: offset ray origins from the cube surface. | Ray 起点相对立方体表面的外偏移（米）。
- `--ray-max-distance <meters>`: max ray query distance. | Ray 最大检测距离（米）。
- `--ray-update-interval <sec>`: sampling interval. | 采样周期（秒）。
- `--force-mapping {spring_damper,legacy}`: force mapping mode (default: `spring_damper`). | 力映射模式（默认：`spring_damper`）。
- `--spring-k <value>`: spring stiffness `k` in N/m (used by `spring_damper`). | 弹簧刚度 `k`，单位 N/m（`spring_damper` 使用）。
- `--damping-c <value>`: damping coefficient `c` in N·s/m (used by `spring_damper`). | 阻尼系数 `c`，单位 N·s/m（`spring_damper` 使用）。
- `--force-max <value>`: max force for `legacy` linear mapping only. | 仅用于 `legacy` 线性映射的最大力参数。
- `--ray-direction {+x,-x}`: ray direction. | Ray 方向。
- `--no-auto-view`: disable auto-launch of external viewer. | 不自动启动外部可视化。

### 2) External viewer script | 外部可视化脚本

```bash
python test/ray_live_view.py [options]
```

Important options | 常用参数:
- `--vmin/--vmax`: force colormap range. | 力图颜色范围。
- `--delta-vmax`: delta colormap max. | delta 图颜色上限。
- `--delta-dot-vmin/--delta-dot-vmax`: delta_dot colormap range. | delta_dot 图颜色范围。
- `--show-distance`: show distance window (off by default). | 显示 distance 窗口（默认关闭）。
- `--interval`: UI refresh interval. | 窗口刷新周期。
- `--show-text`: show numeric values on cells. | 在格子中显示数值。

## Common Examples | 常用命令示例

### A. Default run (auto viewer)
```bash
python test/ray_open_test.py --keep-open --ray-padding 0.0005
```

### B. Open viewer manually (show force + delta + delta_dot)
```bash
python test/ray_live_view.py --vmin 0 --vmax 0.5 --delta-dot-vmin -1 --delta-dot-vmax 1 --cmap inferno --interval 0.1
```

### C. Run with spring-damper force mapping (default)
```bash
python test/ray_open_test.py --keep-open --force-mapping spring_damper --spring-k 1000 --damping-c 5
```

### D. Switch to legacy linear mapping
```bash
python test/ray_open_test.py --keep-open --force-mapping legacy --force-max 1.0
```

### E. Also show distance window
```bash
python test/ray_live_view.py --show-distance --vmin 0 --vmax 0.5 --cmap inferno
```

## Tactile Simulation Principle | 触觉仿真原理（简述）

The system uses a ray grid on the sensor-facing surface. Each frame:
系统在传感面上布置 Ray 网格。每一帧流程如下：

1. Cast rays to get hit distance `distance` (or no hit).  
   发射射线得到命中距离 `distance`（或无命中）。

2. Convert distance to penetration-like quantity `delta`:  
   将距离转换为“压入量” `delta`：  
   `delta = max(0, max_distance - distance)`

3. Estimate penetration velocity `delta_dot`:  
   用离散差分估计侵入速度 `delta_dot`：  
   `delta_dot = (delta_t - delta_{t-1}) / dt`

4. Map `delta`/`delta_dot` to force:  
   将 `delta`/`delta_dot` 映射为力：  
   Default (`spring_damper`, recommended):  
   默认（`spring_damper`，推荐）：  
   `Fn = k * delta + c * delta_dot`, with unilateral clamp `Fn = max(0, Fn)`  
   其中施加单边接触约束：`Fn = max(0, Fn)`  
   Legacy linear mode (`legacy`):  
   旧线性模式（`legacy`）：  
   `force = (delta / max_distance) * force_max`

Notes | 说明:
- `delta = 0` when ray misses or exceeds `max_distance`.  
  无命中或超阈值时 `delta=0`。
- `delta_dot > 0` usually means increasing penetration; `< 0` means release.  
  `delta_dot > 0` 通常表示压入加深，`< 0` 表示回弹/离开。
