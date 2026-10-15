#!/usr/bin/env python3
"""Offline, causal metric RGB feasibility check using the existing backend.

student_inputs.npz contains ONLY RGB, acquisition stamps and CameraInfo.K.
Optional simulator depth is loaded only AFTER all predictions are written.
No truth scale, future frames, or control commands are used.
"""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from realtime_safety.config import ReconstructionConfig
from realtime_safety.types import FramePacket
from realtime_safety.pipeline.video_depth import VideoDepthBackend
from realtime_safety.pipeline.apriltag_calibration import AprilTagScaleCalibrator
from realtime_safety.pipeline.rgb_tag_depth_scale import RgbTagDepthScale


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--calibration',choices=['edges','pose_depth'],default='edges');a=p.parse_args()
    inputs=np.load(a.directory/'student_inputs.npz');k=inputs['camera_k']
    config=ReconstructionConfig(input_size=280,max_points=8000,voxel_size=.01,
        focal_length_x=float(k[0]),focal_length_y=float(k[4]),principal_point_x=float(k[2]),principal_point_y=float(k[5]),
        apriltag_enabled=True,apriltag_id=0,apriltag_size_m=.08,apriltag_detection_interval=1,apriltag_warmup_detections=3)
    backend=VideoDepthBackend(config);backend.load();backend.warmup()
    calibrator=AprilTagScaleCalibrator(config) if a.calibration=='edges' else RgbTagDepthScale();predictions=[];rows=[]
    for i,(rgb,stamp) in enumerate(zip(inputs['rgb'],inputs['stamps'])):
        bgr=rgb[...,::-1].copy();packet=FramePacket(i,float(stamp)*1e-9,time.perf_counter(),bgr,rgb,0.,rgb.shape[1],rgb.shape[0])
        started=time.perf_counter();cloud=backend.infer(packet)
        if a.calibration=='edges': cloud=calibrator.calibrate(bgr,cloud)
        else:
            scale=calibrator.update(rgb,cloud.pointmap[...,1],k,float(stamp)*1e-9)
            if scale is not None: cloud.pointmap*=scale
            cloud.apriltag_locked=scale is not None
        predictions.append(cloud.pointmap[...,1].copy())
        rows.append(dict(frame=i,stamp=float(stamp)*1e-9,processing_ms=(time.perf_counter()-started)*1000,scale=calibrator.scale,tag_locked=cloud.apriltag_locked))
    backend.close()
    suffix='' if a.calibration=='edges' else '_pose_depth'
    np.savez_compressed(a.directory/f'rgb_estimated_depth{suffix}.npz',depth=np.asarray(predictions),stamps=inputs['stamps'])
    # Evaluation boundary: truth has not been opened by inference/calibration.
    truth=np.load(a.directory/'evaluation_only_depth.npz')
    for row,pred,actual in zip(rows,predictions,truth['depth']):
        reference=cv2.resize(actual,(pred.shape[1],pred.shape[0]),interpolation=cv2.INTER_NEAREST)
        valid=np.isfinite(reference)&np.isfinite(pred)&(reference>.05)&(reference<2.)
        if np.any(valid):
            errors=np.abs(pred[valid]-reference[valid]);row.update(evaluation_median_depth_error_m=float(np.median(errors)),evaluation_p95_depth_error_m=float(np.quantile(errors,.95)))
    report=dict(model='existing Metric Video Depth Anything Small',scale_source='RGB AprilTag id 0 with known 0.08m side; no world pose or truth depth',input='one RGB stream + CameraInfo.K',rows=rows,production_ready=False,note='Offline feasibility only. Does not establish obstacle-boundary accuracy, closed-loop safety, or latency under concurrent GUI/tracker load.')
    report['calibration_method']=a.calibration
    (a.directory/f'report{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(frames=len(rows),calibrated_frames=sum(r['tag_locked'] for r in rows),processing_ms_median=float(np.median([r['processing_ms'] for r in rows])),last_frame=rows[-1]),indent=2))
if __name__=='__main__':main()
