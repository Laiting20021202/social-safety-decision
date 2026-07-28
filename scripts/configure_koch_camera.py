#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply bandwidth-safe parameters to the Koch ROS 2 camera")
    parser.add_argument("--node", default="/koach_webcam")
    parser.add_argument("--output-encoding", default="yuv422_yuy2")
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()

    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters

    context = Context()
    rclpy.init(args=[], context=context)
    node = Node("realtime_safety_camera_configurator", context=context)
    service_name = f"{args.node.rstrip('/')}/set_parameters"
    client = node.create_client(SetParameters, service_name)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    deadline = time.monotonic() + max(args.timeout, 0.1)
    try:
        while time.monotonic() < deadline:
            if client.service_is_ready():
                break
            executor.spin_once(timeout_sec=min(0.25, max(deadline - time.monotonic(), 0.0)))
        else:
            print(f"Camera parameter service unavailable: {args.node}")
            return 2

        request = SetParameters.Request(
            parameters=[
                Parameter(
                    name="output_encoding",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_STRING,
                        string_value=args.output_encoding,
                    ),
                )
            ]
        )
        future = client.call_async(request)
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=min(0.25, max(deadline - time.monotonic(), 0.0)))
        if not future.done():
            print(f"Timed out configuring camera: {args.node}")
            return 2
        response = future.result()
        results = response.results if response is not None else []
        if not results or not all(result.successful for result in results):
            reasons = ", ".join(result.reason for result in results if not result.successful)
            print(f"Camera rejected output encoding {args.output_encoding}: {reasons}")
            return 2
        print(f"Configured {args.node} output_encoding={args.output_encoding}")
        return 0
    finally:
        executor.remove_node(node)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        context.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
