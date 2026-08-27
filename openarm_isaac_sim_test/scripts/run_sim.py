#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ros2_ws/src/openarm_isaac_bridge"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OpenArm sorting safety simulation")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mode", choices=("ground_truth", "perception"), default="ground_truth")
    parser.add_argument("--scenario", default="no_obstacle")
    parser.add_argument("--no-ros", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until the app closes")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument(
        "--report-contacts",
        action="store_true",
        help="Print unique PhysX contact body pairs (diagnostic)",
    )
    parser.add_argument("--report-gains", action="store_true", help="Print articulation gains")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    bridge = None
    exit_code = 0
    try:
        import numpy as np

        from openarm_sim.config import load_yaml
        from openarm_sim.isaac_scene import IsaacSceneBuilder, save_stage
        from openarm_sim.scenario import (
            HandTrajectory,
            MotionPhase,
            ScenarioSample,
            bounded_step,
            load_scenario,
        )

        builder = IsaacSceneBuilder()
        built = builder.build()
        contact_reporter = None
        contact_body_paths: list[str] = []
        contact_pairs: set[tuple[str, str]] = set()
        if args.report_contacts:
            contact_reporter, contact_body_paths = _enable_contact_reports(built)
            print(f"CONTACT_REPORT_BODIES {len(contact_body_paths)}", flush=True)
            built.world.reset()
            builder._set_robot_home(built.robot)
        active_scenario = args.scenario
        hand_trajectory = HandTrajectory(load_scenario(active_scenario))
        # Manual reset/withdraw always returns to the shared off-table park,
        # independent of whichever scripted scenario was selected last.
        manual_park = np.asarray(
            HandTrajectory(load_scenario("no_obstacle")).parked_position,
            dtype=float,
        )
        physics_dt = float(load_yaml("config/scene.yaml")["physics"]["dt"])
        camera_config = load_yaml("config/camera.yaml")["camera"]
        # The configured rate is the production default.  A lower override is
        # useful for motion-only smoke tests on machines where RTX RGB-D readback
        # dominates wall time; it never changes timestamps or the camera model.
        camera_fps = float(os.environ.get("OPENARM_CAMERA_FPS", camera_config["fps"]))
        if camera_fps <= 0.0:
            raise ValueError("OPENARM_CAMERA_FPS must be greater than zero")
        camera_period_steps = max(1, round(1.0 / (physics_dt * camera_fps)))
        if not args.no_ros:
            from openarm_isaac_bridge.runtime import IsaacRosBridge

            bridge = IsaacRosBridge(
                mode=args.mode,
                camera=built.camera,
                robot=built.robot,
                hand=built.hand,
                camera_config=camera_config,
                robot_config=load_yaml("config/openarm.yaml")["robot"],
                hand_config=load_yaml("config/hand_scenarios.yaml")["defaults"],
                world=built.world,
                cube_specs=built.cube_specs,
            )
        step = 0
        task_state = "HOME"
        scenario_time = 0.0
        scenario_paused = False
        last_sample = hand_trajectory.sample(0.0, task_state=task_state)
        manual_enabled = False
        manual_speed = 0.3
        manual_target = np.asarray(last_sample.position, dtype=float)
        workspace_min = np.asarray(
            load_yaml("config/hand_scenarios.yaml")["defaults"]["manual_workspace_min"],
            dtype=float,
        )
        workspace_max = np.asarray(
            load_yaml("config/hand_scenarios.yaml")["defaults"]["manual_workspace_max"],
            dtype=float,
        )
        last_hand_phase = last_sample.phase
        local_commands: list[str] = []
        ui_window = None if args.headless else _create_ui(local_commands)
        while simulation_app.is_running() and (args.max_steps <= 0 or step < args.max_steps):
            sim_time = step * physics_dt
            commands = list(local_commands)
            local_commands.clear()
            if bridge is not None:
                commands.extend(bridge.consume_hand_commands())
                target = bridge.consume_manual_hand_target()
                if target is not None:
                    manual_target = np.clip(
                        np.asarray(target, dtype=float), workspace_min, workspace_max
                    )
            for command in commands:
                parts = command.lower().split(":", maxsplit=1)
                action = parts[0]
                if action == "scenario" and len(parts) == 2:
                    active_scenario = parts[1]
                    hand_trajectory = HandTrajectory(load_scenario(active_scenario))
                    scenario_time = 0.0
                    manual_enabled = False
                    manual_target = np.asarray(hand_trajectory.parked_position)
                elif action == "manual" and len(parts) == 2:
                    manual_enabled = parts[1] == "on"
                elif action == "speed" and len(parts) == 2:
                    requested_speed = float(parts[1])
                    if requested_speed not in {0.1, 0.3, 0.6}:
                        raise ValueError(f"unsupported manual hand speed: {requested_speed}")
                    manual_speed = requested_speed
                elif action in {"start", "trigger", "trigger_hand"}:
                    hand_trajectory.trigger(scenario_time)
                elif action == "withdraw":
                    if manual_enabled:
                        manual_target = manual_park.copy()
                    else:
                        hand_trajectory.withdraw(scenario_time, last_sample.position)
                elif action == "reset_hand":
                    manual_enabled = True
                    manual_target = manual_park.copy()
                elif action == "pause":
                    scenario_paused = True
                elif action == "resume":
                    scenario_paused = False
                elif action == "reset":
                    built.world.reset()
                    hand_trajectory = HandTrajectory(load_scenario(active_scenario))
                    scenario_time = 0.0
                    manual_enabled = False
                    manual_target = np.asarray(hand_trajectory.parked_position)
            if not scenario_paused:
                scenario_time += physics_dt
            if manual_enabled:
                position = bounded_step(
                    last_sample.position, manual_target, manual_speed, physics_dt
                )
                moving = not np.allclose(position, manual_target, atol=1e-6)
                sample = ScenarioSample(
                    position,
                    MotionPhase.MOVING_IN if moving else MotionPhase.HOLDING,
                    0,
                )
            else:
                sample = hand_trajectory.sample(scenario_time, task_state=task_state)
            if bridge is not None and sample.phase is not last_hand_phase:
                event = {
                    MotionPhase.MOVING_IN: "hand_motion_started",
                    MotionPhase.HOLDING: "hand_entered",
                    MotionPhase.MOVING_OUT: "hand_withdraw_started",
                    MotionPhase.COMPLETE: "hand_withdrawn",
                }.get(sample.phase)
                if event is not None:
                    bridge.publish_event(event, active_scenario, str(sample.cycle))
                last_hand_phase = sample.phase
            last_sample = sample
            built.hand.set_world_pose(position=np.asarray(sample.position))
            if bridge is not None:
                bridge.update(sim_time, publish_camera=step % camera_period_steps == 0)
                task_state = bridge.task_state
            built.world.step(render=step % camera_period_steps == 0)
            # Isaac's fast headless loop can otherwise monopolize the Python
            # interpreter between sparse render frames.  Yield briefly so the
            # in-process rclpy executor can accept/cancel action goals before
            # simulated time races ahead of the controller.
            if bridge is not None:
                time.sleep(0.001)
            if contact_reporter is not None:
                contact_pairs.update(_collect_contact_pairs(contact_reporter, contact_body_paths))
            step += 1
        if args.export:
            print(f"SCENE_EXPORT_OK={save_stage(args.export)}")
        if args.capture_dir:
            _capture_rgbd(built.camera, args.capture_dir)
        for spec in built.cube_specs:
            cube = built.world.scene.get_object(spec.name)
            if cube is not None:
                position, _ = cube.get_world_pose()
                print(f"CUBE_FINAL {spec.name} {np.asarray(position).tolist()}", flush=True)
        joint_positions = built.robot.get_joint_positions()
        joint_velocities = built.robot.get_joint_velocities()
        print(
            "ROBOT_FINAL "
            f"dofs={len(built.robot.dof_names)} "
            f"finite={bool(np.isfinite(joint_positions).all()) if joint_positions is not None else False} "
            f"max_abs_velocity={float(np.max(np.abs(joint_velocities))) if joint_velocities is not None else 'none'}",
            flush=True,
        )
        if joint_positions is not None and joint_velocities is not None:
            applied = built.robot.get_applied_action()
            measured = built.robot.get_measured_joint_efforts()
            for name, position, velocity in zip(
                built.robot.dof_names, joint_positions, joint_velocities, strict=True
            ):
                print(
                    f"JOINT_FINAL {name} position={float(position):.6f} "
                    f"velocity={float(velocity):.6f} "
                    f"target={float(applied.joint_positions[list(built.robot.dof_names).index(name)]):.6f} "
                    f"effort={float(measured[list(built.robot.dof_names).index(name)]):.6f}",
                    flush=True,
                )
        if args.report_gains:
            controller = built.robot.get_articulation_controller()
            gains = controller.get_gains()
            print(f"ROBOT_GAINS kps={np.asarray(gains[0]).tolist()}", flush=True)
            print(f"ROBOT_GAINS kds={np.asarray(gains[1]).tolist()}", flush=True)
            print(
                f"ROBOT_MAX_EFFORTS {np.asarray(controller.get_max_efforts()).tolist()}",
                flush=True,
            )
        if contact_reporter is not None:
            _print_contact_pairs(contact_pairs)
        print(
            f"SIM_RUN_OK steps={step} mode={args.mode} scenario={args.scenario}", flush=True
        )
        _ = ui_window
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        if bridge is not None:
            bridge.close()
        simulation_app.close()
    return exit_code


def _capture_rgbd(camera: object, directory: Path) -> None:
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    rgba = camera.get_rgba()
    depth = camera.get_depth()
    if rgba is None or depth is None:
        raise RuntimeError("camera did not produce RGB-D frames")
    Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(directory / "rgb.png")
    np.save(directory / "depth_m.npy", np.asarray(depth, dtype=np.float32))


def _enable_contact_reports(built: object) -> tuple[object, list[str]]:
    from isaacsim.sensors.physics import _sensor
    from pxr import PhysxSchema, Usd, UsdPhysics

    paths: list[str] = []
    articulation_root = "/World/OpenArm/root_joint"
    for prim in Usd.PrimRange.Stage(
        built.world.stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        if not str(prim.GetPath()).startswith(articulation_root):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report.CreateThresholdAttr(0.0)
        paths.append(str(prim.GetPath()))
    return _sensor.acquire_contact_sensor_interface(), paths


def _collect_contact_pairs(
    reporter: object, body_paths: list[str]
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in body_paths:
        for contact in reporter.get_rigid_body_raw_data(path):
            body0 = reporter.decode_body_name(contact["body0"])
            body1 = reporter.decode_body_name(contact["body1"])
            pairs.add(tuple(sorted((body0, body1))))
    return pairs


def _print_contact_pairs(pairs: set[tuple[str, str]]) -> None:
    for body0, body1 in sorted(pairs):
        print(f"CONTACT_PAIR {body0} {body1}", flush=True)


def _create_ui(command_queue: list[str]) -> object:
    import omni.ui as ui

    window = ui.Window("OpenArm Safety Controls", width=260, height=230)
    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("Deterministic hand scenario")
            for label, command in (
                ("Reset", "reset"),
                ("Start", "start"),
                ("Pause", "pause"),
                ("Resume", "resume"),
                ("Trigger hand", "trigger_hand"),
                ("Withdraw hand", "withdraw"),
            ):
                ui.Button(label, clicked_fn=lambda value=command: command_queue.append(value))
    return window


if __name__ == "__main__":
    raise SystemExit(main())
