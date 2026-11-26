"""ROS 2 camera relay. Each DDS domain uses its own transport profile."""
import multiprocessing
import os

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import serialize_message
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_msgs.msg import TFMessage


STREAMS = {
    "color/image_raw": Image,
    "color/image_raw/compressed": CompressedImage,
    "color/camera_info": CameraInfo,
}
LIVE = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
STATIC = QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def receive(send_pipe, stopping):
    # camera3 uses the stock Fast DDS UDP fragment size. A separate DDS
    # process lets the gateway receive it and send smaller packets over Wi-Fi.
    os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = "/opt/right-curl/camera-source.xml"
    rclpy.init(domain_id=4)
    reader = rclpy.create_node("laiting_d435_reader")
    last_slot = {}

    def forward(msg, topic):
        # Limit uncompressed RGB on Wi-Fi. Keep full-rate calibration for both transports.
        slot = Time.from_msg(msg.header.stamp).nanoseconds // 100_000_000
        if topic == "color/image_raw" and last_slot.get(topic) == slot:
            return
        last_slot[topic] = slot
        send_pipe.send((topic, serialize_message(msg)))

    for topic, cls in STREAMS.items():
        reader.create_subscription(cls, "/camera/camera3/" + topic,
                                   lambda msg, name=topic: forward(msg, name), LIVE)
    transforms = {}

    def forward_transforms(msg):
        changed = [t for t in msg.transforms if t.child_frame_id.startswith("camera3_")]
        if changed:
            transforms.update((t.child_frame_id, t) for t in changed)
            send_pipe.send(("/tf_static", serialize_message(TFMessage(transforms=list(transforms.values())))))

    reader.create_subscription(TFMessage, "/tf_static", forward_transforms, STATIC)
    try:
        while rclpy.ok() and not stopping.is_set():
            rclpy.spin_once(reader, timeout_sec=0.1)
    finally:
        reader.destroy_node()
        send_pipe.close()
        rclpy.try_shutdown()


def main():
    if int(os.environ["ROS_DOMAIN_ID"]) == 4:
        raise SystemExit("The gateway domain must differ from the D435 source domain")
    context = multiprocessing.get_context("spawn")
    receive_pipe, send_pipe = context.Pipe(duplex=False)
    stopping = context.Event()
    source = context.Process(target=receive, args=(send_pipe, stopping))
    source.start()
    send_pipe.close()
    rclpy.init()
    writer = rclpy.create_node("d435_relay")
    outputs = {name: writer.create_publisher(cls, "/camera/d435/" + name, LIVE)
               for name, cls in STREAMS.items()}
    outputs["/tf_static"] = writer.create_publisher(TFMessage, "/tf_static", STATIC)
    try:
        while rclpy.ok() and source.is_alive():
            if receive_pipe.poll(0.2):
                name, cdr = receive_pipe.recv()
                outputs[name].publish(cdr)
    finally:
        stopping.set()
        receive_pipe.close()
        source.join(timeout=1.0)
        if source.is_alive():
            source.terminate()
            source.join()
        writer.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
