#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("OPENARM_SIM_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and export the deterministic OpenArm stage")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "worlds/openarm_sorting.usd")
    parser.add_argument("--inspect-collisions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    exit_code = 0
    try:
        from openarm_sim.isaac_scene import IsaacSceneBuilder, save_stage

        print("SCENE_BUILD_START", flush=True)
        built = IsaacSceneBuilder().build()
        print("SCENE_BUILD_COMPLETE", flush=True)
        if args.inspect_collisions:
            from pxr import Usd, UsdPhysics

            print(f"ROBOT_PRIM {built.robot.prim.GetPath()}", flush=True)
            for prim in Usd.PrimRange.Stage(
                built.world.stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
            ):
                if not str(prim.GetPath()).startswith("/World/OpenArm"):
                    continue
                print(f"ROBOT_STAGE_PRIM {prim.GetPath()} {prim.GetTypeName()}", flush=True)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    print(f"ROBOT_COLLIDER {prim.GetPath()}", flush=True)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    print(f"ROBOT_BODY {prim.GetPath()}", flush=True)
        for _ in range(args.warmup_frames):
            built.world.step(render=True)
        output = save_stage(args.output)
        print(f"SCENE_EXPORT_OK={output}", flush=True)
        print(f"OPENARM_ASSET={built.robot_asset.kind}:{built.robot_asset.path}", flush=True)
        print(f"CUBE_COUNT={len(built.cube_specs)}", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
