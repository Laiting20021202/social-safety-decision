"""Metric monocular scale from RGB tag pose and calibrated pinhole intrinsics.

Only known printed side length, camera intrinsics and current/past RGB/depth
estimates enter this calibrator. It does not consume simulator depth or poses.
"""
from __future__ import annotations
import cv2
import numpy as np


class RgbTagDepthScale:
    def __init__(self, side_m=.08, tag_id=0, maximum_reprojection_error_px=1.5):
        self.side_m=float(side_m); self.tag_id=int(tag_id)
        self.maximum_reprojection_error_px=float(maximum_reprojection_error_px)
        self.detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),cv2.aruco.DetectorParameters())
        self.scale=None;self.last_stamp=None;self.reprojection_error=None

    def update(self, rgb, estimated_depth, camera_k, stamp):
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return self.scale
        corners,ids,_=self.detector.detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        if ids is None or self.tag_id not in ids: return self.scale
        pixel=np.asarray(corners[int(np.flatnonzero(ids.reshape(-1)==self.tag_id)[0])],np.float64).reshape(4,2)
        return self.update_corners(pixel, estimated_depth, camera_k, rgb.shape[:2], stamp)

    def update_corners(self, pixels, estimated_depth, camera_k, source_shape, stamp):
        """Public geometry path also supports another RGB fiducial detector."""
        if self.last_stamp is not None and stamp <= self.last_stamp: return self.scale
        half=self.side_m/2
        object_points=np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]],np.float64)
        k=np.asarray(camera_k,np.float64).reshape(3,3)
        ok,rvec,tvec=cv2.solvePnP(object_points,np.asarray(pixels,np.float64),k,np.zeros(5),flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or float(tvec[2,0]) <= 0: return self.scale
        projected,_=cv2.projectPoints(object_points,rvec,tvec,k,np.zeros(5))
        error=float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-pixels)**2,axis=1))))
        if error > self.maximum_reprojection_error_px: return self.scale
        # Interior samples reduce depth-edge artifacts; each has its own metric
        # plane depth from the RGB pose, which matters during camera rotation.
        h,w=estimated_depth.shape;sh,sw=source_shape
        mapped=np.asarray(pixels)*[w/sw,h/sh]
        polygon=mapped.mean(axis=0)+(mapped-mapped.mean(axis=0))*.55
        mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.round(polygon).astype(np.int32),1)
        y,x=np.nonzero(mask); model=np.asarray(estimated_depth)[y,x]
        rays=np.column_stack([(x+.5)*sw/w-.5,(y+.5)*sh/h-.5,np.ones(len(x))]) @ np.linalg.inv(k).T
        rotation,_=cv2.Rodrigues(rvec); normal=rotation[:,2]
        denominator=rays@normal
        metric=(normal@tvec[:,0])/denominator
        valid=np.isfinite(model)&(model>.05)&np.isfinite(metric)&(metric>.05)
        if valid.sum()<6: return self.scale
        candidate=float(np.median(metric[valid]/model[valid]))
        if not .02<candidate<50: return self.scale
        if self.scale is None:self.scale=candidate
        else:self.scale=.8*self.scale+.2*float(np.clip(candidate,self.scale*.8,self.scale*1.2))
        self.last_stamp=float(stamp);self.reprojection_error=error
        return self.scale
