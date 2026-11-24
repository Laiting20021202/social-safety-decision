from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from realtime_safety.ros2_bridge.image_subscriber import (
    compressed_image_message_to_bgr,
    image_message_to_bgr,
)


def _message(image: np.ndarray, encoding: str, step: int | None = None):
    height, width = image.shape[:2]
    return SimpleNamespace(
        height=height,
        width=width,
        step=step or int(image.strides[0]),
        encoding=encoding,
        data=image.tobytes(),
    )


def test_rgb8_image_message_converts_to_bgr() -> None:
    rgb = np.array([[[255, 20, 3], [7, 80, 190]]], dtype=np.uint8)

    bgr = image_message_to_bgr(_message(rgb, "rgb8"))

    np.testing.assert_array_equal(bgr, rgb[..., ::-1])


def test_yuyv_image_message_converts_to_bgr() -> None:
    yuyv = np.array([[[80, 128], [120, 128]]], dtype=np.uint8)
    expected = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

    bgr = image_message_to_bgr(_message(yuyv, "yuv422_yuy2"))

    np.testing.assert_array_equal(bgr, expected)


def test_compressed_image_message_decodes_jpeg_to_bgr() -> None:
    bgr = np.zeros((12, 16, 3), dtype=np.uint8)
    bgr[:, :8] = (10, 40, 220)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    message = SimpleNamespace(data=encoded.tobytes(), format="bgr8; jpeg compressed bgr8")

    decoded = compressed_image_message_to_bgr(message)

    assert decoded.shape == bgr.shape
    assert decoded.dtype == np.uint8
    # JPEG is lossy; compare the two broad color regions rather than pixels.
    np.testing.assert_allclose(decoded[:, :8].mean(axis=(0, 1)), bgr[:, :8].mean(axis=(0, 1)), atol=8)
