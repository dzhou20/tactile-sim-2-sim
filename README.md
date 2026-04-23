# tactile-sim-2-sim

## Quick Start | 快速开始

Run from repo root:

```bash
python test/ray_open_test.py --keep-open --ray-padding 0.0005
```

This starts Isaac Sim ray sampling and writes logs (`distance`, `delta`, `delta_dot`, `force`) to `data/ray_logs/` under the repo root.
该命令启动 Isaac Sim 射线采样，并在仓库根目录下的 `data/ray_logs/` 写出日志（`distance`、`delta`、`delta_dot`、`force`）。

## Environment Setup | 运行环境配置

### Validated Environment | 已验证环境

The code in this repository is currently known to run in the following environment:
这个仓库当前已知可运行的环境如下：

- Conda environment name: `env_visualtac`
- Conda 环境名：`env_visualtac`
- Python: `3.11.15`
- Isaac Sim: `5.1.0.0`
- Isaac Lab: `0.54.3`
- `isaaclab-contrib`: `0.0.2`
- PyTorch: `2.7.0+cu128`
- NumPy: `1.26.0`
- Matplotlib: `3.10.3`

In general, this repository should run in environments with Isaac Sim `>= 4.5.0`.
总体上，这个仓库在 Isaac Sim `>= 4.5.0` 的运行环境中都应该可以工作。

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
- `--target-contact-body-path <prim_path>`: target rigid body to track continuously for taxel contact state (default: `/World/Cylinder_Test`). | 用于持续跟踪 taxel 接触状态的目标刚体 prim path（默认：`/World/Cylinder_Test`）。
- `--no-auto-view`: disable auto-launch of external viewer. | 不自动启动外部可视化。

### Target Body Tracking | 目标物体持续跟踪

The current contact-state logic in `test/ray_open_test.py` tracks a specific target rigid body. By default, the target is `/World/Cylinder_Test`, and it can be overridden with `--target-contact-body-path`.
当前 `test/ray_open_test.py` 里的接触状态逻辑会持续跟踪一个特定的目标刚体。默认目标是 `/World/Cylinder_Test`，也可以通过 `--target-contact-body-path` 覆盖。

This is intentional: the tracker is not only checking whether a ray hit something in the current frame. It is maintaining a continuous contact episode for the same object across frames.
这不是单纯判断“这一帧有没有命中某个物体”，而是为了在跨帧过程中持续维护“同一个物体上的连续接触 episode”。

Why this target is tracked continuously:
为什么要持续跟踪这个目标物体：
- Only hits on the target body are treated as valid contact for the taxel contact state machine.
- 只有命中目标刚体时，taxel contact 状态机才会认为这是有效接触。
- On contact start, the code stores the same world contact point in both the sensor local frame and the target-body local frame.
- 在接触开始时，代码会把同一个世界系接触点同时记录到传感器局部坐标系和目标物体局部坐标系中。
- During later frames of the same contact episode, those anchors are reused to compute relative motion in the sensor frame, including `d_A`, `d_t_A`, `v_t_A`, `xi_t`, and `F_t_candidate`.
- 在同一个接触 episode 的后续帧里，会复用这些锚点来计算传感器坐标系下的相对运动，包括 `d_A`、`d_t_A`、`v_t_A`、`xi_t` 和 `F_t_candidate`。

Practical implication:
实际含义：
- If the configured target body path does not exist in the loaded USD stage, the target-contact tracker will not report meaningful target contact states.
- 如果配置的目标刚体路径在当前加载的 USD 场景里不存在，那么目标接触跟踪器就不会产出有意义的目标接触状态。
- If rays hit another object, that hit may still appear in raw ray results, but it will not be treated as tracked target contact by this logic.
- 如果 Ray 打到了别的物体，这些命中仍可能出现在原始 ray 结果里，但不会被这套逻辑当作“持续跟踪的目标接触”。

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

## Temporal Considerations | 时序相关的潜在误差 

### Discrete sampling and velocity estimation | 离散采样与速度估计误差

All time-derivative quantities (`delta_dot`, `v_t_A`) are computed by finite difference over the sampling interval `dt`:  
所有时间导数量（`delta_dot`、`v_t_A`）均通过采样间隔 `dt` 的有限差分计算：

```
delta_dot = (delta[i] - delta[i-1]) / dt
```

When the underlying signal is not smoothly differentiable — e.g., at the moment of contact onset or during fast impacts — this discrete approximation introduces transient errors proportional to `1/dt`. The error is bounded in practice because `d_t_A` is derived from rigid-body transforms (not raw ray hits), and the default `dt = 0.1 s` is large enough to avoid noise amplification.  
当底层信号不可连续微分时（如接触瞬间或快速碰撞），离散近似引入的瞬态误差与 `1/dt` 成正比。由于 `d_t_A` 来自刚体变换而非原始射线命中点，且默认 `dt = 0.1 s` 足够大，噪声放大效应在实践中是有界的。

For reference, typical sampling rates:  
常见采样频率参考：
- Isaac Sim physics: ~60–200 Hz (dt ≈ 5–17 ms)  
  Isaac Sim 物理仿真：~60–200 Hz（dt ≈ 5–17 ms）
- Real tactile sensors (e.g. DIGIT, GelSight): 30–60 Hz (dt ≈ 17–33 ms)  
  真实触觉传感器（如 DIGIT、GelSight）：30–60 Hz（dt ≈ 17–33 ms）
- This script default: 10 Hz (dt = 100 ms) — conservative, reduces noise sensitivity  
  本脚本默认：10 Hz（dt = 100 ms）——偏保守，降低噪声敏感性

### Damping term during slow contact | 缓慢接触时阻尼项的影响

The damping contribution `c * v_t_A` is proportional to contact velocity. During slow, deliberate manipulation — which is typical in dexterous grasping tasks — contact velocities are low (order of 0.01–0.05 m/s), so the damping force is small relative to the spring term `k * d_t_A`. This aligns with the practical emphasis in manipulation research on **policy robustness over peak-speed grasping**: policies trained or evaluated at low contact speeds are less sensitive to damping coefficient tuning, and sim-to-real transfer of the tangential force signal is more reliable in this regime.  
阻尼项 `c * v_t_A` 与接触速度成正比。在灵巧抓取任务中，接触运动通常缓慢且受控（速度量级约 0.01–0.05 m/s），阻尼力相对弹簧项 `k * d_t_A` 较小。这与 manipulation 研究中**优先策略鲁棒性而非极速抓取**的导向一致：在低接触速度下训练或评估的策略对阻尼系数的敏感性更低，切向力信号的 sim-to-real 迁移在此区间也更可靠。
