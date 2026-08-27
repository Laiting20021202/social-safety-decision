from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assets import RobotAsset, resolve_robot_asset
from .camera_math import camera_aim_direction, camera_world_position, intrinsics_from_horizontal_fov
from .config import PROJECT_ROOT, load_yaml
from .scene_model import CubeSpec, deterministic_cube_layout


@dataclass
class BuiltScene:
    world: Any
    camera: Any
    robot: Any
    hand: Any
    cube_specs: list[CubeSpec]
    robot_asset: RobotAsset


class IsaacSceneBuilder:
    """Build the complete stage after ``SimulationApp`` has been initialized."""

    def __init__(self) -> None:
        self.scene_config = load_yaml("config/scene.yaml")
        self.camera_config = load_yaml("config/camera.yaml")["camera"]
        self.robot_config = load_yaml("config/openarm.yaml")["robot"]
        self.hand_config = load_yaml("config/hand_scenarios.yaml")["defaults"]

    def build(self) -> BuiltScene:
        from isaacsim.core.api import World

        physics = self.scene_config["physics"]
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=float(physics["dt"]),
            rendering_dt=float(physics["rendering_dt"]),
        )
        world.scene.add_default_ground_plane(z_position=0.0)
        self._configure_physics(world)
        self._add_table(world)
        self._add_bins(world)
        cube_specs = self._add_cubes(world)
        self._add_calibration_marker(world)
        hand = self._add_hand_obstacle()
        robot_asset = resolve_robot_asset(self.robot_config)
        robot = self._add_robot(world, robot_asset)
        camera = self._add_camera()
        self._add_lighting()
        world.reset()
        self._set_robot_home(robot)
        camera.initialize()
        self._configure_camera(camera)
        camera.add_distance_to_image_plane_to_frame()
        return BuiltScene(world, camera, robot, hand, cube_specs, robot_asset)

    def _set_robot_home(self, robot: Any) -> None:
        """Apply a gravity-stable default before the first user-visible physics step."""

        names = list(robot.dof_names)
        positions = np.asarray(robot.get_joint_positions(), dtype=float)
        kps = np.full(len(names), 2000.0, dtype=float)
        kds = np.full(len(names), 100.0, dtype=float)
        max_efforts = np.full(len(names), 333.0, dtype=float)
        # Official OpenArm Isaac Lab HIGH_PD values. Gravity remains enabled;
        # HIGH_PD's gravity-disable shortcut is intentionally not copied.
        arm_kps = [400.0] * 7
        arm_kds = [80.0] * 7
        arm_efforts = [40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0]
        for side in ("left", "right"):
            joint_names = self.robot_config["joint_names"][side]
            for offset, joint_name in enumerate(joint_names):
                index = names.index(joint_name)
                positions[index] = float(self.robot_config["home"][side][offset])
                kps[index] = arm_kps[offset]
                kds[index] = arm_kds[offset]
                max_efforts[index] = arm_efforts[offset]
        for index, name in enumerate(names):
            if "finger_joint" in name:
                positions[index] = float(self.robot_config["gripper"]["open_position"])
        zeros = np.zeros_like(positions)
        robot.set_joints_default_state(positions=positions, velocities=zeros, efforts=zeros)
        robot.set_joint_positions(positions)
        robot.set_joint_velocities(zeros)
        robot.set_solver_position_iteration_count(16)
        robot.set_solver_velocity_iteration_count(4)
        controller = robot.get_articulation_controller()
        controller.set_gains(kps=kps, kds=kds)
        controller.set_max_efforts(max_efforts)
        from isaacsim.core.utils.types import ArticulationAction

        controller.apply_action(ArticulationAction(joint_positions=positions))

    def _configure_physics(self, world: Any) -> None:
        from pxr import Gf, PhysxSchema, UsdPhysics

        physics = self.scene_config["physics"]
        stage = world.stage
        scene_prim = stage.GetPrimAtPath("/physicsScene")
        if not scene_prim.IsValid():
            scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        else:
            scene = UsdPhysics.Scene(scene_prim)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(abs(float(physics["gravity"][2])))
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physx_scene.CreateEnableGPUDynamicsAttr(bool(physics["gpu_dynamics"]))

    def _add_table(self, world: Any) -> None:
        from isaacsim.core.api.objects import FixedCuboid

        table = self.scene_config["table"]
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Workstation/Table",
                name="table",
                position=np.asarray(table["center"], dtype=float),
                scale=np.asarray(table["size"], dtype=float),
                size=1.0,
                color=np.asarray(table["color_rgb"], dtype=float),
            )
        )
        edge = table["yellow_edge"]
        table_center = np.asarray(table["center"], dtype=float)
        table_size = np.asarray(table["size"], dtype=float)
        edge_position = np.array(
            [
                table_center[0],
                table_center[1] - table_size[1] / 2.0 + float(edge["width"]) / 2.0,
                float(table["top_z"]) + float(edge["height"]) / 2.0,
            ]
        )
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Workstation/YellowEdge",
                name="yellow_table_edge",
                position=edge_position,
                scale=np.array([table_size[0], edge["width"], edge["height"]], dtype=float),
                size=1.0,
                color=np.asarray(edge["color_rgb"], dtype=float),
            )
        )

    def _add_bins(self, world: Any) -> None:
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.utils.semantics import add_update_semantics

        bins = self.scene_config["bins"]
        inner = np.asarray(bins["inner_size"], dtype=float)
        wall = float(bins["wall_thickness"])
        base = float(bins["base_thickness"])
        colors = self.scene_config["cubes"]["colors"]
        for color_name, center_value in bins["centers"].items():
            center = np.asarray(center_value, dtype=float)
            tint = np.asarray(colors[color_name], dtype=float) * 0.65 + 0.25
            pieces = {
                "base": (center + [0.0, 0.0, -inner[2] / 2.0], [inner[0] + 2 * wall, inner[1] + 2 * wall, base]),
                "front": (center + [-inner[0] / 2.0 - wall / 2.0, 0.0, 0.0], [wall, inner[1] + 2 * wall, inner[2]]),
                "back": (center + [inner[0] / 2.0 + wall / 2.0, 0.0, 0.0], [wall, inner[1] + 2 * wall, inner[2]]),
                "left": (center + [0.0, inner[1] / 2.0 + wall / 2.0, 0.0], [inner[0], wall, inner[2]]),
                "right": (center + [0.0, -inner[1] / 2.0 - wall / 2.0, 0.0], [inner[0], wall, inner[2]]),
            }
            for piece_name, (position, scale) in pieces.items():
                piece = world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/Bins/{color_name}/{piece_name}",
                        name=f"{color_name}_bin_{piece_name}",
                        position=np.asarray(position, dtype=float),
                        scale=np.asarray(scale, dtype=float),
                        size=1.0,
                        color=tint,
                    )
                )
                add_update_semantics(piece.prim, f"{color_name}_sorting_bin")

    def _add_cubes(self, world: Any) -> list[CubeSpec]:
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.semantics import add_update_semantics

        cubes = self.scene_config["cubes"]
        size = float(cubes["size"])
        specs = deterministic_cube_layout(self.scene_config)
        for spec in specs:
            cube = world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cubes/{spec.name}",
                    name=spec.name,
                    position=np.asarray(spec.position, dtype=float),
                    scale=np.full(3, size),
                    size=1.0,
                    color=np.asarray(cubes["colors"][spec.color], dtype=float),
                    mass=float(cubes["mass_kg"]),
                )
            )
            add_update_semantics(cube.prim, f"{spec.color}_sorting_cube")
        return specs

    def _add_calibration_marker(self, world: Any) -> None:
        marker = self.scene_config["apriltag"]
        if not marker["enabled"]:
            return
        from isaacsim.core.api.objects import VisualCuboid

        import cv2

        size = float(marker["size"])
        center = np.asarray(marker["center"], dtype=float)
        marker_pixels = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
            int(marker["id"]),
            8,
            borderBits=1,
        )
        # A 1 cm white quiet zone keeps the black AprilTag border visible on
        # the dark tabletop while the encoded marker itself remains 0.08 m.
        world.scene.add(
            VisualCuboid(
                prim_path="/World/CalibrationMarker/quiet_zone",
                name="calibration_quiet_zone",
                position=center + np.array([0.0, 0.0, 0.00025]),
                scale=np.array([size + 0.02, size + 0.02, 0.0005]),
                size=1.0,
                color=np.ones(3),
            )
        )
        cell = size / float(marker_pixels.shape[0])
        for row in range(marker_pixels.shape[0]):
            for col in range(marker_pixels.shape[1]):
                position = center + np.array(
                    [
                        (col - (marker_pixels.shape[1] - 1) / 2.0) * cell,
                        ((marker_pixels.shape[0] - 1) / 2.0 - row) * cell,
                        0.0005,
                    ],
                    dtype=float,
                )
                world.scene.add(
                    VisualCuboid(
                        prim_path=f"/World/CalibrationMarker/cell_{row}_{col}",
                        name=f"calibration_cell_{row}_{col}",
                        position=position,
                        scale=np.array([cell, cell, 0.001]),
                        size=1.0,
                        color=(
                            np.ones(3)
                            if int(marker_pixels[row, col]) > 127
                            else np.zeros(3)
                        ),
                    )
                )

    def _add_hand_obstacle(self) -> Any:
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.semantics import add_update_semantics
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = self._stage()
        root_path = "/World/HandObstacle"
        root = UsdGeom.Xform.Define(stage, root_path)
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        rigid_body.CreateKinematicEnabledAttr(True)

        palm = self.hand_config["collision_proxy"]["palm_size"]
        visual_parts = {
            "palm": ([0.0, 0.0, 0.0], palm),
            "thumb": ([0.015, -0.050, -0.005], [0.065, 0.018, 0.018]),
            "finger_1": ([0.065, 0.030, 0.0], [0.12, 0.015, 0.015]),
            "finger_2": ([0.070, 0.010, 0.0], [0.13, 0.015, 0.015]),
            "finger_3": ([0.066, -0.010, 0.0], [0.12, 0.015, 0.015]),
            "finger_4": ([0.058, -0.030, 0.0], [0.10, 0.015, 0.015]),
            "forearm": ([-0.22, 0.0, 0.0], [0.34, 0.075, 0.075]),
        }
        for name, (offset, scale) in visual_parts.items():
            cube = UsdGeom.Cube.Define(stage, f"{root_path}/Visual/{name}")
            cube.CreateSizeAttr(1.0)
            transform = cube.AddTransformOp()
            transform.Set(
                Gf.Matrix4d().SetScale(Gf.Vec3d(*scale)).SetTranslateOnly(Gf.Vec3d(*offset))
            )
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.70, 0.40, 0.28)])
            add_update_semantics(cube.GetPrim(), "placeholder_human_hand")

        proxy_parts = {
            "palm_box": ([0.0, 0.0, 0.0], palm),
            "wrist_box": ([-0.10, 0.0, 0.0], [0.16, 0.065, 0.065]),
            "forearm_box": ([-0.30, 0.0, 0.0], [0.34, 0.09, 0.09]),
        }
        for name, (offset, scale) in proxy_parts.items():
            proxy = UsdGeom.Cube.Define(stage, f"{root_path}/Collision/{name}")
            proxy.CreateSizeAttr(1.0)
            transform = proxy.AddTransformOp()
            transform.Set(
                Gf.Matrix4d().SetScale(Gf.Vec3d(*scale)).SetTranslateOnly(Gf.Vec3d(*offset))
            )
            UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
            proxy.MakeInvisible()

        hand = SingleXFormPrim(prim_path=root_path, name="hand_obstacle")
        parked = np.asarray(self.hand_config.get("parked_position", [0.30, -0.75, 1.05]))
        hand.set_world_pose(position=parked)
        return hand

    def _add_robot(self, world: Any, robot_asset: RobotAsset) -> Any:
        import omni.kit.commands
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.prims import SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        prim_path = str(self.robot_config["prim_path"])
        if robot_asset.kind == "urdf":
            cached_usd = PROJECT_ROOT / "assets/openarm_cache/openarm_v10_bimanual.usd"
            if not cached_usd.is_file():
                from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

                status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
                if not status:
                    raise RuntimeError("Isaac Sim failed to create an OpenArm URDF import config")
                import_config.merge_fixed_joints = False
                import_config.fix_base = bool(self.robot_config["fix_base"])
                import_config.self_collision = bool(self.robot_config["enabled_self_collisions"])
                import_config.make_default_prim = True
                import_config.create_physics_scene = False
                import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
                import_config.default_drive_strength = 1200.0
                import_config.default_position_drive_damping = 80.0
                status, imported_path = omni.kit.commands.execute(
                    "URDFParseAndImportFile",
                    urdf_path=str(robot_asset.path),
                    import_config=import_config,
                    dest_path=str(cached_usd),
                    get_articulation_root=True,
                )
                if not status or not imported_path:
                    raise RuntimeError(f"OpenArm URDF import failed: {robot_asset.path}")
            asset_path = cached_usd
        else:
            asset_path = robot_asset.path
        add_reference_to_stage(usd_path=str(asset_path), prim_path=prim_path)
        # Pose the reference container itself.  Robot resolves its public prim
        # handle to the imported fixed joint on Isaac Sim 4.5; passing position
        # to Robot would therefore move the joint prim rather than the model.
        reference_xform = SingleXFormPrim(prim_path=prim_path, name="openarm_reference")
        reference_xform.set_world_pose(
            position=np.asarray(self.robot_config["base_position"], dtype=float)
        )
        articulation_path = f"{prim_path}/root_joint"
        self._configure_robot_gravity(world.stage, articulation_path)
        self._apply_srdf_collision_filters(world.stage, articulation_path)
        robot = world.scene.add(
            Robot(
                # The URDF importer's default prim is a reference container.  The
                # actual ArticulationRootAPI lives on its root_joint child.  Point
                # Robot at that child so PhysX drives and link state queries bind
                # to the articulation rather than the reference container.
                prim_path=articulation_path,
                name="openarm",
            )
        )
        return robot

    def _configure_robot_gravity(self, stage: Any, root: str) -> None:
        """Apply the configured gravity mode to OpenArm links only.

        Isaac Lab's high-PD manipulator configuration disables link gravity so
        position drives can track their commanded state without a separate
        gravity-compensation controller. Dynamic cubes and all other scene
        bodies remain gravity enabled.
        """

        from pxr import PhysxSchema, Usd, UsdPhysics

        root_prim = stage.GetPrimAtPath(root)
        if not root_prim.IsValid():
            raise RuntimeError(f"OpenArm articulation root is missing: {root}")
        disable_gravity = not bool(self.robot_config["gravity_enabled"])
        rigid_body_count = 0
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(
                disable_gravity
            )
            rigid_body_count += 1
        if rigid_body_count == 0:
            raise RuntimeError("OpenArm contains no rigid bodies to configure")

    @staticmethod
    def _apply_srdf_collision_filters(stage: Any, root: str) -> None:
        """Keep self-collision enabled while filtering SRDF-adjacent link pairs."""

        from pxr import Sdf, UsdPhysics

        pairs: list[tuple[str, str]] = []
        for side in ("left", "right"):
            pairs.append(("openarm_body_link0", f"openarm_{side}_link0"))
            for index in range(7):
                pairs.append((f"openarm_{side}_link{index}", f"openarm_{side}_link{index + 1}"))
            # The simplified v1.0 collision meshes for links 5 and 7 overlap
            # around their shared link-6 wrist housing at valid joint angles.
            # Treat this two-hop mechanical enclosure like an adjacent pair;
            # all other non-adjacent self-collision pairs remain enabled.
            pairs.append((f"openarm_{side}_link5", f"openarm_{side}_link7"))
            hand = f"openarm_{side}_hand"
            link7 = f"openarm_{side}_link7"
            left_finger = f"openarm_{side}_left_finger"
            right_finger = f"openarm_{side}_right_finger"
            pairs.extend(
                [
                    (hand, link7),
                    (hand, left_finger),
                    (hand, right_finger),
                    (link7, left_finger),
                    (link7, right_finger),
                    (left_finger, right_finger),
                ]
            )
        for first, second in pairs:
            first_prim = stage.GetPrimAtPath(f"{root}/{first}")
            second_path = Sdf.Path(f"{root}/{second}")
            if not first_prim.IsValid() or not stage.GetPrimAtPath(second_path).IsValid():
                raise RuntimeError(f"OpenArm SRDF collision-filter link is missing: {first}, {second}")
            relation = UsdPhysics.FilteredPairsAPI.Apply(first_prim).CreateFilteredPairsRel()
            relation.AddTarget(second_path)

    def _add_camera(self) -> Any:
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera

        camera = self.camera_config
        scene_center = self.scene_config["zones"]["workspace"]["center"]
        position = camera_world_position(
            scene_center,
            float(camera["height_above_table"]),
            float(camera["horizontal_offset_to_workspace_center"]),
            float(camera["lateral_offset"]),
        )
        if bool(camera.get("aim_at_workspace_center", False)):
            target = np.asarray(scene_center, dtype=float)
            target += np.asarray(camera.get("aim_offset", [0.0, 0.0, 0.0]), dtype=float)
            camera_aim_direction(position, target)
        else:
            pitch = math.radians(float(camera["downward_pitch_deg"]))
            yaw = math.radians(float(camera["yaw_deg"]))
            direction = np.array(
                [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)]
            )
            target = position + direction
        width = int(camera["resolution"]["width"])
        height = int(camera["resolution"]["height"])
        sensor = Camera(
            prim_path="/World/Sensors/RGBD/Camera",
            name="rgbd_camera",
            frequency=int(camera["fps"]),
            resolution=(width, height),
        )
        set_camera_view(eye=position, target=target, camera_prim_path=sensor.prim_path)
        return sensor

    def _configure_camera(self, sensor: Any) -> None:
        camera = self.camera_config
        width = int(camera["resolution"]["width"])
        height = int(camera["resolution"]["height"])
        intrinsics = intrinsics_from_horizontal_fov(
            width, height, float(camera["horizontal_fov_deg"])
        )
        horizontal_aperture = 2.0
        sensor.set_horizontal_aperture(horizontal_aperture)
        sensor.set_vertical_aperture(horizontal_aperture * height / width)
        sensor.set_focal_length(horizontal_aperture * intrinsics.fx / width)
        sensor.set_clipping_range(float(camera["near_clip"]), float(camera["far_clip"]))
        sensor.set_projection_type("pinhole")

    def _add_lighting(self) -> None:
        from pxr import Gf, UsdGeom, UsdLux

        stage = self._stage()
        lighting = self.scene_config["lighting"]
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(float(lighting["dome_intensity"]))
        key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
        key.CreateIntensityAttr(float(lighting["key_light_intensity"]))
        orient = UsdGeom.Xformable(key.GetPrim()).AddOrientOp()
        orient.Set(Gf.Quatf(0.9239, Gf.Vec3f(0.2706, -0.2706, 0.0)))

    @staticmethod
    def _stage() -> Any:
        import omni.usd

        return omni.usd.get_context().get_stage()


def save_stage(path: str | Path) -> Path:
    import omni.usd

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    if not stage.Export(str(destination)):
        raise RuntimeError(f"failed to export USD stage: {destination}")
    return destination
