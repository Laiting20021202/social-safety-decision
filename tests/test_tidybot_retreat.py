from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from realtime_safety.pipeline.tidybot_retreat import (
    RetreatConfig, RetreatController, nearest_obstacle, rotation2,
)
from realtime_safety.ros2_bridge.tidybot_retreat import cloud_xyz, load_settings


def cloud(x=.75, y=.05):
    return np.tile([x, y, .6], (40, 1))


def start(yaw=0, obstacle=None, config=None):
    c = RetreatController(config or RetreatConfig())
    for i in range(3):
        t = .1 * i
        c.update_pose([0, 0, yaw], t+1, t)
        c.update_cloud(cloud() if obstacle is None else obstacle, t+1, t)
        velocity = c.tick(t)
    assert c.state == "MOVING"
    return c, np.array(velocity)


@pytest.mark.parametrize("yaw", [0, np.pi/2, -1.3])
@pytest.mark.parametrize("side", [-.3, .3])
def test_feedback_drives_two_meters_opposite_obstacle_then_holds(yaw, side):
    c, velocity = start(yaw, cloud(.7, side))
    direction = rotation2(yaw) @ (-np.array([.7, side]) / np.hypot(.7, side))
    np.testing.assert_allclose(c.target, 2 * direction)
    pose = np.array([0., 0., yaw])
    for i in range(1, 500):
        pose[:2] += rotation2(pose[2]) @ velocity * .05
        t = .2 + i*.05
        c.update_pose(pose, t+1, t)
        c.update_cloud(cloud(), t+1, t)
        velocity = np.array(c.tick(t))
        if c.state == "DONE":
            break
    assert c.state == "DONE", c.reason
    assert np.linalg.norm(pose[:2] - 2*direction) <= .03
    assert np.linalg.norm(pose[:2]) <= 2.0
    assert c.tick(t+.1) == (0, 0)
    c.update_cloud(cloud(), t+2, t+.1)
    assert c.tick(t+.1) == (0, 0)  # Persistent obstacle never triggers another 2 m.


def test_body_command_accounts_for_rotation_and_world_mode():
    body, b = start(np.pi/2, cloud(.75, 0))
    world, w = start(np.pi/2, cloud(.75, 0), replace(RetreatConfig(), command_frame="world"))
    np.testing.assert_allclose(b, [-.15, 0], atol=1e-9)
    np.testing.assert_allclose(w, [0, -.15], atol=1e-9)


def test_requires_three_distinct_observations_not_timer_ticks():
    c=RetreatController(RetreatConfig())
    c.update_pose([0,0,0], 1, 0)
    c.update_cloud(cloud(), 1, 0)
    for t in [.1,.2,.3]:
        c.update_cloud(cloud(), 1, t)
        assert c.tick(t) == (0,0)
    assert c.near_hits == 1


@pytest.mark.parametrize("missing", ["cloud", "pose"])
def test_stale_sensor_latches_zero_and_does_not_auto_restart(missing):
    c,_=start()
    t=2
    if missing == "cloud": c.update_pose([0,0,0], 3, t)
    else: c.update_cloud(cloud(), 3, t)
    assert c.tick(t) == (0,0)
    assert c.state == "FAULT"
    c.update_pose([0,0,0], 4, 2.1)
    c.update_cloud(cloud(), 4, 2.1)
    assert c.tick(2.1) == (0,0)


def test_replayed_joint_stamps_do_not_keep_control_alive():
    c,_=start()
    c.update_pose([0,0,0], 1.2, 1)
    c.update_cloud(cloud(), 2, 1)
    assert c.tick(1) == (0,0)
    assert c.state == "FAULT"


def test_reset_requires_fresh_clear_scene():
    c,_=start()
    c.fault("stop")
    assert not c.reset(.3)
    c.update_cloud(cloud(2),2,.3)
    assert c.reset(.3)
    assert c.tick(.3) == (0,0)


def test_pose_reset_teleport_and_malformed_feedback_stop():
    for pose, stamp in [([0,0,0], .1), ([3,0,0],2), ([float('nan'),0,0],2)]:
        c,_=start()
        c.update_pose(pose,stamp,.3)
        assert c.tick(.3) == (0,0)
        assert c.state == "FAULT"


def test_stall_despite_fresh_data_stops():
    c,_=start()
    for i in range(1,40):
        t=.2+i*.1
        c.update_pose([0,0,0],t+1,t)
        c.update_cloud(cloud(),t+1,t)
        command=c.tick(t)
    assert c.state == "FAULT"
    assert command == (0,0)


def test_floor_robot_points_outliers_and_nonfinite_cloud():
    cfg=RetreatConfig()
    points=np.concatenate([np.tile([.7,0,0],(40,1)),cloud(.1),[[.6,.3,.6]]])
    assert nearest_obstacle(points,cfg) is None
    np.testing.assert_allclose(nearest_obstacle(np.concatenate([points,cloud()]),cfg),[.75,.05])
    with pytest.raises(ValueError): nearest_obstacle(np.full((40,3),np.nan),cfg)


def test_empty_cloud_is_fault_not_clear_space():
    c,_=start()
    c.update_cloud(np.empty((0,3)),2,.3)
    assert c.state == "FAULT"
    assert not c.reset(.3)


@pytest.mark.parametrize("change", [{"speed_mps":1}, {"travel_m":float('nan')},
    {"confirm_frames":2.5}, {"trigger_distance_m":.2}, {"max_duration_s":1}])
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValueError): RetreatConfig(**change)


@pytest.mark.parametrize("big", [False,True])
def test_decode_organized_cloud_with_row_padding(big):
    expected=np.arange(12,dtype=float).reshape(4,3)
    raw=bytearray(64)
    dtype=np.dtype('>f4' if big else '<f4')
    view=np.ndarray((2,2,3),dtype=dtype,buffer=raw,strides=(32,12,4))
    view[:]=expected.reshape(2,2,3)
    msg=SimpleNamespace(fields=[SimpleNamespace(name=k,offset=i*4,datatype=7,count=1) for i,k in enumerate('xyz')],
        is_bigendian=big,height=2,width=2,point_step=12,row_step=32,data=raw)
    np.testing.assert_allclose(cloud_xyz(msg),expected)
    msg.data=raw[:10]
    with pytest.raises(ValueError):cloud_xyz(msg)


def test_execution_refuses_unverified_scale_and_controller(tmp_path):
    path=Path(__file__).resolve().parents[1]/'configs/tidybot_retreat.yaml'
    settings,_,transform=load_settings(str(path),False)
    # Isaac camera is 20 degrees down: optical forward maps forward and down;
    # optical right maps to base right (negative base Y).
    np.testing.assert_allclose(transform @ [0,0,0,1],
        [0.1,0.0102028,1.3149987,1],atol=1e-9)
    np.testing.assert_allclose(transform[:3,:3] @ [1,0,0],[0,-1,0],atol=1e-9)
    np.testing.assert_allclose(transform[:3,:3] @ [0,0,1],
        [np.cos(np.deg2rad(20)),0,-np.sin(np.deg2rad(20))],atol=1e-9)
    with pytest.raises(ValueError,match='metric_scale_verified'):load_settings(str(path),True)
    # Explicit operator acceptance only bypasses the two unresolved demo facts;
    # the configuration continues to report both as unverified.
    accepted,_,_=load_settings(str(path),True,True)
    assert accepted['verified']['metric_scale_verified'] is False
    assert accepted['verified']['retreat_corridor_verified_clear'] is False
    settings['verified']={k:True for k in settings['verified']}
    config=tmp_path/'verified.yaml'
    config.write_text(yaml.safe_dump(settings))
    assert load_settings(str(config),True)[1].travel_m == 2
    settings['cloud_to_base'][0][0]=2
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError,match='rigid'):load_settings(str(config),True)


def test_pose_marks_done_before_late_cloud_can_replace_completion():
    controller = RetreatController(RetreatConfig())
    controller.state = "MOVING"
    controller.pose = np.array([-1.9, 0., 0.])
    controller.pose_at = 0.
    controller.pose_stamp = 1.
    controller.cloud_stamp = 10.
    controller.cloud_at = 0.
    controller.origin = np.array([0., 0.])
    controller.target = np.array([-2., 0.])
    controller.direction = np.array([-1., 0.])
    controller.started_at = controller.progress_at = 0.
    controller.update_pose([-1.98, 0., 0.], 2., 1.)
    assert controller.state == "DONE"
    controller.update_cloud(np.tile([.6, 0., .75], (40, 1)), 9., 1.01)
    assert controller.state == "DONE"
