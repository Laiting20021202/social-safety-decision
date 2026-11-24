from __future__ import annotations

from functools import partial

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class HandObstacleController(Node):
    def __init__(self) -> None:
        super().__init__("hand_obstacle_controller")
        self.declare_parameter("scenario", "no_obstacle")
        self.publisher = self.create_publisher(String, "/sim/hand/command", 10)
        self._services = []
        for command in ("reset", "start", "pause", "resume", "trigger_hand", "withdraw"):
            self._services.append(
                self.create_service(
                    Trigger,
                    f"/sim/hand/{command}",
                    partial(self._service_callback, command=command),
                )
            )
        scenario = str(self.get_parameter("scenario").value)
        self.create_timer(0.5, self._publish_initial_scenario, callback_group=None)
        self._initial_scenario = scenario
        self._scenario_sent = False

    def _publish_initial_scenario(self) -> None:
        if not self._scenario_sent:
            self.publisher.publish(String(data=f"scenario:{self._initial_scenario}"))
            self._scenario_sent = True

    def _service_callback(self, _request: Trigger.Request, response: Trigger.Response, command: str):
        self.publisher.publish(String(data=command))
        response.success = True
        response.message = f"sent {command}"
        return response


def main() -> None:
    rclpy.init()
    node = HandObstacleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
