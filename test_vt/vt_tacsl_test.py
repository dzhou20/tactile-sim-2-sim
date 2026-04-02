"""Open vt_tacsl_test.usd and run a minimal force-field visuo-tactile sensor test."""

import argparse
import logging
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_USD_PATH = REPO_ROOT / "assets" / "isaac_sim" / "vt_tacsl_test.usd"
DEFAULT_DT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal visuo-tactile force-field test on vt_tacsl_test.usd")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--keep-open", action="store_true", help="Keep running until Ctrl+C")
    parser.add_argument("--steps", type=int, default=120, help="Number of simulation steps when not using --keep-open")
    parser.add_argument(
        "--usd-path",
        type=Path,
        default=DEFAULT_USD_PATH,
        help="USD file to open. Defaults to assets/isaac_sim/vt_tacsl_test.usd.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=SCRIPT_DIR / "vt_tacsl_test.log",
        help="Path to the runtime log file. Defaults to vt_tacsl_test.log next to this script.",
    )
    parser.add_argument(
        "--experience",
        type=str,
        default="python",
        help=(
            "Experience preset or .kit path to launch. "
            "Supported presets: full, python, base, default. "
            "Defaults to python for stability in headless mode."
        ),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def configure_logging(log_file: Path) -> logging.Logger:
    log_file = log_file.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("vt_tacsl_test")
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
        return str((apps_dir / candidate_path).resolve())
    if candidate_path.suffix == ".kit" and candidate_path.exists():
        return str(candidate_path.resolve())
    if candidate in preset_to_file.values():
        return str((apps_dir / candidate).resolve())

    raise ValueError(
        f"Unsupported --experience value: {experience!r}. "
        "Use one of: full, python, base, default, or pass a valid .kit path."
    )


def open_stage(ctx, usd_path: Path, simulation_app, logger: logging.Logger):
    usd_path = usd_path.resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    logger.info("Opening USD stage: %s", usd_path)
    opened = ctx.open_stage(str(usd_path))
    if not opened:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    if hasattr(ctx, "is_loading"):
        while ctx.is_loading() and simulation_app.is_running():
            simulation_app.update()
    else:
        for _ in range(120):
            simulation_app.update()

    for _ in range(240):
        stage = ctx.get_stage()
        if stage is not None:
            logger.info("USD stage loaded successfully")
            return stage
        simulation_app.update()

    raise RuntimeError(f"USD stage did not become available: {usd_path}")


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_file)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    from isaacsim import SimulationApp

    experience_file = resolve_experience_file(args.experience)
    simulation_app_cfg = {"headless": args.headless}
    logger.info(
        "Parsed args: headless=%s keep_open=%s steps=%s usd_path=%s experience=%s log_file=%s",
        args.headless,
        args.keep_open,
        args.steps,
        args.usd_path,
        args.experience,
        args.log_file,
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
        from pxr import PhysxSchema, UsdGeom, UsdPhysics

        import isaaclab.sim as sim_utils
        from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg
        from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg

        torch.set_printoptions(precision=6, sci_mode=False)

        def _ensure_bool_attr(attr, create_attr_fn, value: bool) -> None:
            if attr.IsValid():
                attr.Set(value)
            else:
                create_attr_fn(value)

        def _get_aligned_local_bounds(prim):
            bbox_cache = UsdGeom.BBoxCache(
                time=0.0,
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            aligned_box = bbox_cache.ComputeLocalBound(prim).ComputeAlignedBox()
            min_pt = aligned_box.GetMin()
            max_pt = aligned_box.GetMax()
            return (
                (float(min_pt[0]), float(min_pt[1]), float(min_pt[2])),
                (float(max_pt[0]), float(max_pt[1]), float(max_pt[2])),
            )

        def _create_box_mesh(
            prim_path: str,
            min_pt: tuple[float, float, float],
            max_pt: tuple[float, float, float],
            *,
            invisible: bool,
        ) -> None:
            min_x, min_y, min_z = min_pt
            max_x, max_y, max_z = max_pt
            mesh = UsdGeom.Mesh.Define(stage, prim_path)
            mesh.CreateSubdivisionSchemeAttr().Set("none")
            mesh.CreatePointsAttr(
                [
                    (min_x, min_y, min_z),
                    (max_x, min_y, min_z),
                    (max_x, max_y, min_z),
                    (min_x, max_y, min_z),
                    (min_x, min_y, max_z),
                    (max_x, min_y, max_z),
                    (max_x, max_y, max_z),
                    (min_x, max_y, max_z),
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
            mesh.CreateExtentAttr([min_pt, max_pt])
            if invisible:
                UsdGeom.Imageable(mesh).MakeInvisible()

        def _prepare_scene_for_force_field() -> str:
            sensor_prim_path = "/World/Elastomer/TacSLSensor"
            elastomer_prim = stage.GetPrimAtPath("/World/Elastomer")
            contact_object_prim = stage.GetPrimAtPath("/World/ContactObject")
            if not elastomer_prim.IsValid():
                raise RuntimeError("Missing required prim: /World/Elastomer")
            if not contact_object_prim.IsValid():
                raise RuntimeError("Missing required prim: /World/ContactObject")

            elastomer_collision_api = UsdPhysics.CollisionAPI(elastomer_prim)
            if not elastomer_collision_api:
                elastomer_collision_api = UsdPhysics.CollisionAPI.Apply(elastomer_prim)
                logger.info("Applied UsdPhysics.CollisionAPI to /World/Elastomer")
            _ensure_bool_attr(
                elastomer_collision_api.GetCollisionEnabledAttr(),
                elastomer_collision_api.CreateCollisionEnabledAttr,
                True,
            )
            rigid_body_api = UsdPhysics.RigidBodyAPI(elastomer_prim)
            if not rigid_body_api:
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(elastomer_prim)
                logger.info("Applied UsdPhysics.RigidBodyAPI to /World/Elastomer")
            _ensure_bool_attr(rigid_body_api.GetRigidBodyEnabledAttr(), rigid_body_api.CreateRigidBodyEnabledAttr, True)
            _ensure_bool_attr(rigid_body_api.GetKinematicEnabledAttr(), rigid_body_api.CreateKinematicEnabledAttr, True)
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(elastomer_prim)
            if not physx_rigid_body_api:
                physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(elastomer_prim)
                logger.info("Applied PhysxSchema.PhysxRigidBodyAPI to /World/Elastomer")
            _ensure_bool_attr(
                physx_rigid_body_api.GetDisableGravityAttr(),
                physx_rigid_body_api.CreateDisableGravityAttr,
                True,
            )

            def _is_visual_mesh(prim) -> bool:
                return prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)

            if not stage.GetPrimAtPath(sensor_prim_path).IsValid():
                UsdGeom.Xform.Define(stage, sensor_prim_path)
                logger.info("Created sensor prim: %s", sensor_prim_path)

            visual_mesh = sim_utils.get_first_matching_child_prim("/World/Elastomer", _is_visual_mesh)
            if visual_mesh is None:
                visual_mesh_path = "/World/Elastomer/TacSLVisualMesh"
                min_pt, max_pt = _get_aligned_local_bounds(elastomer_prim)
                _create_box_mesh(visual_mesh_path, min_pt, max_pt, invisible=True)
                logger.info("Created hidden visual mesh for tactile-point generation: %s", visual_mesh_path)

            # Keep the root contact prim as the rigid body parent, and attach a dedicated SDF mesh under it.
            if contact_object_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                contact_object_prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                logger.info("Removed UsdPhysics.MeshCollisionAPI from /World/ContactObject root prim")
            if contact_object_prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                contact_object_prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
                logger.info("Removed PhysxSchema.PhysxSDFMeshCollisionAPI from /World/ContactObject root prim")
            if contact_object_prim.HasAPI(UsdPhysics.CollisionAPI):
                contact_object_prim.RemoveAPI(UsdPhysics.CollisionAPI)
                logger.info("Removed UsdPhysics.CollisionAPI from /World/ContactObject root prim to avoid duplicate colliders")

            contact_rigid_body_api = UsdPhysics.RigidBodyAPI(contact_object_prim)
            if not contact_rigid_body_api:
                contact_rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(contact_object_prim)
                logger.info("Applied UsdPhysics.RigidBodyAPI to /World/ContactObject")
            _ensure_bool_attr(
                contact_rigid_body_api.GetRigidBodyEnabledAttr(),
                contact_rigid_body_api.CreateRigidBodyEnabledAttr,
                True,
            )
            _ensure_bool_attr(
                contact_rigid_body_api.GetKinematicEnabledAttr(),
                contact_rigid_body_api.CreateKinematicEnabledAttr,
                False,
            )
            contact_physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(contact_object_prim)
            if not contact_physx_rigid_body_api:
                contact_physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(contact_object_prim)
                logger.info("Applied PhysxSchema.PhysxRigidBodyAPI to /World/ContactObject")
            _ensure_bool_attr(
                contact_physx_rigid_body_api.GetDisableGravityAttr(),
                contact_physx_rigid_body_api.CreateDisableGravityAttr,
                False,
            )

            sdf_mesh_path = "/World/ContactObject/TacSLSdfMesh"
            sdf_mesh_prim = stage.GetPrimAtPath(sdf_mesh_path)
            if not sdf_mesh_prim.IsValid():
                min_pt, max_pt = _get_aligned_local_bounds(contact_object_prim)
                _create_box_mesh(sdf_mesh_path, min_pt, max_pt, invisible=True)
                sdf_mesh_prim = stage.GetPrimAtPath(sdf_mesh_path)
                logger.info("Created hidden SDF mesh for contact object: %s", sdf_mesh_path)

            sdf_collision_api = UsdPhysics.CollisionAPI(sdf_mesh_prim)
            if not sdf_collision_api:
                sdf_collision_api = UsdPhysics.CollisionAPI.Apply(sdf_mesh_prim)
                logger.info("Applied UsdPhysics.CollisionAPI to %s", sdf_mesh_path)
            _ensure_bool_attr(
                sdf_collision_api.GetCollisionEnabledAttr(),
                sdf_collision_api.CreateCollisionEnabledAttr,
                True,
            )
            if not sdf_mesh_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(sdf_mesh_prim)
                logger.info("Applied PhysxSchema.PhysxCollisionAPI to %s", sdf_mesh_path)
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(sdf_mesh_prim)
            if not mesh_collision_api:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(sdf_mesh_prim)
                logger.info("Applied UsdPhysics.MeshCollisionAPI to %s", sdf_mesh_path)
            approximation_attr = mesh_collision_api.GetApproximationAttr()
            if approximation_attr.IsValid():
                approximation_attr.Set("sdf")
            else:
                mesh_collision_api.CreateApproximationAttr().Set("sdf")
            if not sdf_mesh_prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(sdf_mesh_prim)
                logger.info("Applied PhysxSchema.PhysxSDFMeshCollisionAPI to %s", sdf_mesh_path)

            logger.info("Prepared contact object rigid body root: /World/ContactObject")
            logger.info("Prepared contact object SDF mesh child: %s", sdf_mesh_path)
            return sensor_prim_path

        ctx = omni.usd.get_context()
        stage = open_stage(ctx, args.usd_path, simulation_app, logger)
        logger.info("Opened stage root layer: %s", stage.GetRootLayer().identifier)
        sensor_prim_path = _prepare_scene_for_force_field()

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DEFAULT_DT))
        if not args.headless:
            sim.set_camera_view(eye=[0.20, 0.20, 0.16], target=[0.0, 0.0, 0.02])

        # The local config class still requires a render_cfg object even when camera tactile is disabled.
        placeholder_render_cfg = GelSightRenderCfg(
            base_data_path=str(Path.home() / "IsaacLab"),
            sensor_data_dir_name="gelsight_r15_data",
            image_height=320,
            image_width=240,
            mm_per_pixel=0.0877,
        )
        sensor_cfg = VisuoTactileSensorCfg(
            prim_path=sensor_prim_path,
            update_period=0.0,
            debug_vis=False,
            enable_camera_tactile=False,
            enable_force_field=True,
            camera_cfg=None,
            render_cfg=placeholder_render_cfg,
            tactile_array_size=(8, 8),
            tactile_margin=0.001,
            contact_object_prim_path_expr="/World/ContactObject",
            normal_contact_stiffness=1.0,
            tangential_stiffness=0.1,
            friction_coefficient=2.0,
        )
        logger.info("Creating VisuoTactileSensor with cfg: %s", sensor_cfg)
        sensor = VisuoTactileSensor(cfg=sensor_cfg)

        sim.reset()
        sensor.reset()
        dt = sim.get_physics_dt()
        logger.info("Simulation reset complete. physics_dt=%.6f", dt)
        logger.info("VisuoTactileSensor prim path: %s", sensor_prim_path)
        logger.info("Elastomer rigid-body parent path: /World/Elastomer")
        logger.info("Contact object root path: /World/ContactObject")
        if not args.headless:
            sim.pause()
            logger.info(
                "Simulation is paused after initialization. Use the Isaac Sim Play button to start physics manually."
            )

        def _log_penetration(step_index: int) -> None:
            data = sensor.data.penetration_depth
            if data is None:
                logger.info("[STEP %04d] penetration_depth=None", step_index)
                return
            env0 = data[0].detach().cpu()
            max_penetration = float(env0.max().item()) if env0.numel() > 0 else 0.0
            logger.info("[STEP %04d] max_penetration=%.6f", step_index, max_penetration)
            if env0.numel() == 64:
                logger.info("[STEP %04d] penetration_depth[0]=\n%s", step_index, env0.reshape(8, 8))
            else:
                logger.info("[STEP %04d] penetration_depth[0]=%s", step_index, env0)

        step_index = 0

        def _wait_for_manual_play_if_needed() -> bool:
            if args.headless or sim.is_playing():
                return True
            simulation_app.update()
            return sim.is_playing()

        if args.keep_open:
            logger.info("Entering keep-open loop. Press Ctrl+C to stop.")
            while simulation_app.is_running():
                if not _wait_for_manual_play_if_needed():
                    continue
                sim.step(render=not args.headless)
                sensor.update(dt, force_recompute=True)
                step_index += 1
                _log_penetration(step_index)
        else:
            if args.headless:
                logger.info("Running for %s simulation steps", args.steps)
            else:
                logger.info(
                    "Waiting for manual Play in the GUI, then running for %s simulation steps.",
                    args.steps,
                )
            while simulation_app.is_running() and step_index < args.steps:
                if not _wait_for_manual_play_if_needed():
                    continue
                sim.step(render=not args.headless)
                sensor.update(dt, force_recompute=True)
                step_index += 1
                _log_penetration(step_index)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, closing Isaac Sim...")
    except Exception:
        logger.exception("vt_tacsl_test failed")
        raise
    finally:
        logger.info("Closing SimulationApp")
        simulation_app.close()


if __name__ == "__main__":
    main()
