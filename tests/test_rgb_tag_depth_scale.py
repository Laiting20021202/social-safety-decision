import cv2
import numpy as np
import pytest
from realtime_safety.pipeline.rgb_tag_depth_scale import RgbTagDepthScale


@pytest.mark.parametrize('pitch',[0.,.5,-.5])
def test_rgb_pose_scale_tracks_camera_angle_without_truth_depth_input(pitch):
    k=np.array([[400.,0,320],[0,400.,240],[0,0,1]])
    points=np.array([[-.04,.04,0],[.04,.04,0],[.04,-.04,0],[-.04,-.04,0]])
    rvec=np.array([pitch,.1,0]);tvec=np.array([.02,.01,.8]);rot,_=cv2.Rodrigues(rvec)
    pixels,_=cv2.projectPoints(points,rvec,tvec,k,None)
    y,x=np.indices((480,640));rays=np.stack([(x-320)/400,(y-240)/400,np.ones_like(x)],axis=-1)
    # Deliberately biased monocular units, to test scale geometry only.
    model=(rot[:,2]@tvec)/(rays@rot[:,2])*3.
    c=RgbTagDepthScale()
    scale=c.update_corners(pixels.reshape(4,2),model,k,(480,640),1.)
    assert scale == pytest.approx(1/3,abs=.002)
    assert c.update_corners(pixels.reshape(4,2),model*2,k,(480,640),.5)==scale
