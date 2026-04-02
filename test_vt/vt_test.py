"""Standalone Isaac Lab visuo-tactile force-field test scene."""

import argparse
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True

DT = 0.01
ELASTOMER_HALF_EXTENTS = (0.05, 0.05, 0.005)
CONTACT_OBJECT_HALF_EXTENTS = (0.02, 0.02, 0.02)
ENV_SPACING = 0.3
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Lab visuo-tactile force-field test")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--keep-open", action="store_true", help="Keep the simulation running until Ctrl+C")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps when not using --keep-open")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=SCRIPT_DIR / "vt_test.log",
        help="Path to the runtime log file. Defaults to vt_test.log next to this script.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Write step statistics every N simulation steps. Defaults to 1.",
    )
    parser.add_argument(
        "--experience",
        type=str,
        default="full",
        help=(
            "Experience preset or .kit path to launch. "
            "Supported presets: full, python, base, default. "
            "Defaults to full."
        ),
    )
    parser.add_argument(
        "--build-test-scene",
        action="store_true",
        help=(
            "Procedurally build the demo elastomer/contact-object scene. "
            "By default the script waits for user-created prims instead."
        ),
    )
    parser.add_argument(
        "--sensor-prim-path",
        type=str,
        default="/World/Elastomer/TactileSensor",
        help=(
            "Sensor prim path to use in manual-scene mode. "
            "Ignored when --build-test-scene is enabled."
        ),
    )
    parser.add_argument(
        "--contact-object-path",
        type=str,
        default="/World/ContactObject",
        help=(
            "Contact object prim path/expression to use in manual-scene mode. "
            "Ignored when --build-test-scene is enabled."
        ),
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> logging.Logger:
    log_file = log_file.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("vt_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def resolve_experience_file(experience: str) -> str | None:
    experience = experience.strip()
    if experience == "" or experience == "default":
        return None

    import isaacsim

    apps_dir = Path(isaacsim.__file__).resolve().parent / "apps"
    preset_to_file = {
        "full": "isaacsim.exp.full.kit",
        "python": "isaacsim.exp.base.python.kit",
        "base": "isaacsim.exp.base.kit",
    }

    candidate = preset_to_file.get(experience, experience)
    candidate_path = Path(candidate)

    if candidate_path.is_absolute():
        return str(candidate_path)
    if candidate_path.suffix == ".kit" and (apps_dir / candidate_path).exists():
        return str(apps_dir / candidate_path)
    if candidate_path.suffix == ".kit" and candidate_path.exists():
        return str(candidate_path.resolve())
    if candidate in preset_to_file.values():
        return str((apps_dir / candidate).resolve())

    raise ValueError(
        f"Unsupported --experience value: {experience!r}. "
        "Use one of: full, python, base, default, or pass a valid .kit path."
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    if args.num_envs < 1:
        raise ValueError("--num_envs must be >= 1")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    from isaacsim import SimulationApp

    experience_file = resolve_experience_file(args.experience)
    simulation_app_cfg = {"headless": args.headless}
    logger.info(
        "Parsed args: headless=%s keep_open=%s num_envs=%s steps=%s experience=%s log_file=%s log_every=%s",
        args.headless,
        args.keep_open,
        args.num_envs,
        args.steps,
        args.experience,
        args.log_file,
        args.log_every,
    )
    logger.info(
        "Scene mode: %s",
        "procedural test scene" if args.build_test_scene else "manual scene",
    )
    if experience_file is None:
        logger.info("Launching Isaac Sim with default experience selection")
        simulation_app = SimulationApp(simulation_app_cfg)
    else:
        logger.info("Launching Isaac Sim with experience: %s", experience_file)
        simulation_app = SimulationApp(simulation_app_cfg, experience=experience_file)

    try:
        import torch
        import omni.usd
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        import isaaclab.sim as sim_utils
        from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg
        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg

        def _set_xform_pose(prim_path: str, translate: tuple[float, float, float]) -> None:
            xform = UsdGeom.Xform.Define(stage, prim_path)
            UsdGeom.XformCommonAPI(xform).SetTranslate(Gf.Vec3d(*translate))

        def _create_box_mesh(
            prim_path: str,
            half_extents: tuple[float, float, float],
            color: tuple[float, float, float],
            collision: bool = False,
            collision_approximation: str | None = None,
        ):
            hx, hy, hz = half_extents
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            mesh.CreateSubdivisionSchemeAttr().Set("none")
            mesh.CreatePointsAttr(
                [
                    (-hx, -hy, -hz),
                    (hx, -hy, -hz),
                    (hx, hy, -hz),
                    (-hx, hy, -hz),
                    (-hx, -hy, hz),
                    (hx, -hy, hz),
                    (hx, hy, hz),
                    (-hx, hy, hz),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([3] * 12)
            mesh.CreateFaceVertexIndicesAttr(
                [
                    0, 1, 2,
                    0, 2, 3,
                    4, 6, 5,
                    4, 7, 6,
                    0, 4, 5,
                    0, 5, 1,
                    3, 2, 6,
                    3, 6, 7,
                    0, 3, 7,
                    0, 7, 4,
                    1, 5, 6,
                    1, 6, 2,
                ]
            )
            mesh.CreateExtentAttr([(-hx, -hy, -hz), (hx, hy, hz)])
            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            prim = mesh.GetPrim()

            if collision:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(prim)
                mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
                if not mesh_collision:
                    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                if collision_approximation is not None:
                    approximation_attr = mesh_collision.GetApproximationAttr()
                    if approximation_attr.IsValid():
                        approximation_attr.Set(collision_approximation)
                    else:
                        mesh_collision.CreateApproximationAttr().Set(collision_approximation)
                    if collision_approximation == "sdf" and not prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            return prim

        def _apply_rigid_body(
            prim_path: str,
            *,
            mass: float | None,
            disable_gravity: bool,
            kinematic: bool,
        ) -> None:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"Invalid prim for rigid body: {prim_path}")

            rb_api = UsdPhysics.RigidBodyAPI(prim)
            if not rb_api:
                rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            enabled_attr = rb_api.GetRigidBodyEnabledAttr()
            if enabled_attr.IsValid():
                enabled_attr.Set(True)
            else:
                rb_api.CreateRigidBodyEnabledAttr(True)
            kinematic_attr = rb_api.GetKinematicEnabledAttr()
            if kinematic_attr.IsValid():
                kinematic_attr.Set(kinematic)
            else:
                rb_api.CreateKinematicEnabledAttr(kinematic)

            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not physx_rb_api:
                physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            disable_gravity_attr = physx_rb_api.GetDisableGravityAttr()
            if disable_gravity_attr.IsValid():
                disable_gravity_attr.Set(disable_gravity)
            else:
                physx_rb_api.CreateDisableGravityAttr(disable_gravity)

            if mass is not None:
                mass_api = UsdPhysics.MassAPI(prim)
                if not mass_api:
                    mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_attr = mass_api.GetMassAttr()
                if mass_attr.IsValid():
                    mass_attr.Set(mass)
                else:
                    mass_api.CreateMassAttr(mass)

        def _set_env_offset(index: int, total_envs: int) -> tuple[float, float, float]:
            x_offset = (index - 0.5 * (total_envs - 1)) * ENV_SPACING
            return (x_offset, 0.0, 0.0)

        def _build_env(root_path: str, env_offset: tuple[float, float, float]) -> None:
            logger.info("Building environment at %s with offset=%s", root_path, env_offset)
            _set_xform_pose(root_path, env_offset)

            elastomer_path = f"{root_path}/Elastomer"
            contact_object_path = f"{root_path}/ContactObject"

            _set_xform_pose(elastomer_path, (0.0, 0.0, 0.05))
            _create_box_mesh(
                f"{elastomer_path}/visual",
                ELASTOMER_HALF_EXTENTS,
                color=(0.85, 0.25, 0.25),
                collision=False,
            )
            _create_box_mesh(
                f"{elastomer_path}/collision",
                ELASTOMER_HALF_EXTENTS,
                color=(0.85, 0.25, 0.25),
                collision=True,
                collision_approximation="convexHull",
            )
            _apply_rigid_body(elastomer_path, mass=None, disable_gravity=True, kinematic=True)
            UsdGeom.Xform.Define(stage, f"{elastomer_path}/TactileSensor")

            _set_xform_pose(contact_object_path, (0.0, 0.0, 0.2))
            _create_box_mesh(
                f"{contact_object_path}/visual",
                CONTACT_OBJECT_HALF_EXTENTS,
                color=(0.2, 0.35, 0.9),
                collision=False,
            )
            _create_box_mesh(
                f"{contact_object_path}/collision",
                CONTACT_OBJECT_HALF_EXTENTS,
                color=(0.2, 0.35, 0.9),
                collision=True,
                collision_approximation="sdf",
            )
            _apply_rigid_body(contact_object_path, mass=0.05, disable_gravity=False, kinematic=False)

        def _get_sensor_paths(num_envs: int) -> tuple[str, str]:
            if num_envs == 1:
                return "/World/Elastomer/TactileSensor", "/World/ContactObject"
            return "/World/envs/env_.*/Elastomer/TactileSensor", "/World/envs/env_.*/ContactObject"

        def _make_sensor_cfg(sensor_prim_path: str, contact_object_expr: str) -> VisuoTactileSensorCfg:
            # Current Isaac Lab API still expects a render config object even when camera tactile is disabled.
            dummy_render_cfg = GelSightRenderCfg(
                sensor_data_dir_name="unused",
                image_height=1,
                image_width=1,
                mm_per_pixel=1.0,
            )
            return VisuoTactileSensorCfg(
                prim_path=sensor_prim_path,
                update_period=0.0,
                debug_vis=False,
                render_cfg=dummy_render_cfg,
                enable_camera_tactile=False,
                enable_force_field=True,
                tactile_array_size=(8, 8),
                tactile_margin=0.002,
                contact_object_prim_path_expr=contact_object_expr,
                normal_contact_stiffness=1.0,
                tangential_stiffness=0.1,
                friction_coefficient=2.0,
                camera_cfg=None,
            )

        def _summarize_tactile_data(sensor: VisuoTactileSensor) -> tuple[float, float, float]:
            data = sensor.data
            if data.tactile_normal_force is None or data.tactile_shear_force is None or data.penetration_depth is None:
                return 0.0, 0.0, 0.0
            max_normal = float(torch.max(data.tactile_normal_force).item())
            shear_magnitude = torch.linalg.norm(data.tactile_shear_force, dim=-1)
            max_tangential = float(torch.max(shear_magnitude).item())
            max_penetration = float(torch.max(data.penetration_depth).item())
            return max_normal, max_tangential, max_penetration

        def _wait_for_prim(stage, prim_path: str) -> None:
            logger.info("Waiting for prim to appear: %s", prim_path)
            while simulation_app.is_running():
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    logger.info("Detected prim: %s", prim_path)
                    return
                simulation_app.update()
            raise RuntimeError(f"Simulation app closed while waiting for prim: {prim_path}")

        sim_utils.create_new_stage()
        logger.info("Created new USD stage")
        sim_cfg = sim_utils.SimulationCfg(dt=DT)
        sim = sim_utils.SimulationContext(sim_cfg)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Failed to acquire USD stage")

        UsdGeom.Xform.Define(stage, "/World")
        if args.build_test_scene:
            physicsUtils.add_ground_plane(
                stage,
                "/World/GroundPlane",
                "Z",
                2.0,
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(0.5, 0.5, 0.5),
            )

            if args.num_envs == 1:
                _build_env("/World", (0.0, 0.0, 0.0))
            else:
                UsdGeom.Xform.Define(stage, "/World/envs")
                for env_index in range(args.num_envs):
                    _build_env(f"/World/envs/env_{env_index}", _set_env_offset(env_index, args.num_envs))
            sensor_prim_path, contact_object_expr = _get_sensor_paths(args.num_envs)
        else:
            if args.num_envs != 1:
                raise ValueError("Manual-scene mode currently supports only --num_envs 1")
            sensor_prim_path = args.sensor_prim_path
            contact_object_expr = args.contact_object_path

        sim_utils.update_stage()

        if not args.headless:
            sim.set_camera_view(eye=(0.45, -0.45, 0.30), target=(0.0, 0.0, 0.05))
            logger.info("Configured camera view for GUI mode")

        if not args.build_test_scene:
            logger.info(
                "Manual-scene mode active. Create the following prims in the stage, then the test will continue: "
                "sensor=%s contact=%s",
                sensor_prim_path,
                contact_object_expr,
            )
            _wait_for_prim(stage, sensor_prim_path)
            _wait_for_prim(stage, contact_object_expr)

        sensor_cfg = _make_sensor_cfg(sensor_prim_path, contact_object_expr)
        tactile_sensor = VisuoTactileSensor(cfg=sensor_cfg)

        logger.info("Simulation config: num_envs=%s dt=%s steps=%s keep_open=%s log_every=%s", args.num_envs, DT, args.steps, args.keep_open, args.log_every)
        logger.info("Sensor prim path: %s", sensor_cfg.prim_path)
        logger.info("Contact object expr: %s", sensor_cfg.contact_object_prim_path_expr)

        sim.reset()
        tactile_sensor.reset()
        logger.info("Simulation and tactile sensor reset complete")

        step_count = 0
        try:
            while simulation_app.is_running():
                if (not args.keep_open) and step_count >= args.steps:
                    logger.info("Reached requested step limit: %s", args.steps)
                    break

                sim.step(render=not args.headless)
                step_count += 1
                tactile_sensor.update(DT, force_recompute=True)

                if step_count % args.log_every == 0:
                    max_normal, max_tangential, max_penetration = _summarize_tactile_data(tactile_sensor)
                    logger.info(
                        "[STEP %04d] max_normal=%.6f max_tangential=%.6f max_penetration=%.6f",
                        step_count,
                        max_normal,
                        max_tangential,
                        max_penetration,
                    )
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, closing Isaac Sim...")

    finally:
        logger.info("Closing SimulationApp")
        simulation_app.close()


if __name__ == "__main__":
    main()
