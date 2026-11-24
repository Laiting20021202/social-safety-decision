import cv2
import numpy as np

from realtime_safety.pipeline.camera_motion import CameraMotionEstimator


def test_affine_camera_translation_is_estimated() -> None:
    rng = np.random.default_rng(4)
    first = np.zeros((160, 220, 3), dtype=np.uint8)
    for x, y in rng.integers([15, 15], [205, 145], size=(80, 2)):
        cv2.circle(first, (int(x), int(y)), 2, (255, 255, 255), -1)
    transform = np.float32([[1, 0, 4], [0, 1, 3]])
    second = cv2.warpAffine(first, transform, (220, 160))
    estimator = CameraMotionEstimator()
    estimator.update(first)
    motion = estimator.update(second)
    assert motion.tracked_points >= 8
    assert motion.confidence > 0.7
    assert np.isclose(motion.affine_2d[0, 2], 4, atol=0.8)
    assert np.isclose(motion.affine_2d[1, 2], 3, atol=0.8)
