#!/usr/bin/env python3
"""Read-only real-time observer + actual onboard Image recorder.

No control publications, synthetic renders, or policy inputs. Wall-clock CFR
preserves actual playback speed. JSONL records RGB acquisition stamp and age
for every frame; observer and onboard RGB use simulation acquisition stamps. Neither stream
is synthesized. The observer stream is recording-only, never a policy input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


def fit(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    result = np.zeros((height, width, 3), np.uint8)
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return result


class Recorder(Node):
    def __init__(self, image_topic, observer_topic):
        super().__init__('openarm_video_recorder')
        self.rgb = None
        self.observer = None
        self.observer_stamp = None
        self.rgb_stamp = None
        self.sim_time = None
        self.state = {}
        self.create_subscription(Image, image_topic, self.image, qos_profile_sensor_data)
        self.create_subscription(Image, observer_topic, lambda msg: self.image(msg, observer=True), qos_profile_sensor_data)
        self.create_subscription(Clock, '/clock', self.clock, qos_profile_sensor_data)
        for topic, name in [('/openarm/task/state', 'task'), ('/openarm/safety/state', 'safety'), ('/openarm/safety/memory_status', 'memory')]:
            self.create_subscription(String, topic, lambda m, name=name: self.state.update({name:m.data}), 10)
    def clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec*1e-9
    def image(self, msg, observer=False):
        channels = 4 if msg.encoding in ('rgba8','bgra8') else 3
        if msg.encoding not in ('rgb8','bgr8','rgba8','bgra8'): return
        raw = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        rgb = raw[:, :msg.width*channels].reshape(msg.height,msg.width,channels)[...,:3]
        if msg.encoding.startswith('rgb'): rgb = rgb[...,::-1]
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        if observer:
            self.observer, self.observer_stamp = rgb.copy(), stamp
        else:
            self.rgb, self.rgb_stamp = rgb.copy(), stamp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=180)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--image-topic', default='/rgbd/color/image_raw')
    p.add_argument('--observer-topic', default='/demo/observer/recording_camera/image_raw')
    p.add_argument('--label', default='Geometric avoidance + visibility memory | RGB-D baseline')
    a=p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init(); n=Recorder(a.image_topic, a.observer_topic)
    thread=threading.Thread(target=rclpy.spin,args=(n,),daemon=True);thread.start()
    deadline = time.monotonic()+15
    while (n.rgb is None or n.observer is None) and time.monotonic()<deadline: time.sleep(.05)
    if n.rgb is None or n.observer is None:
        rclpy.shutdown(); thread.join(timeout=2)
        raise RuntimeError('Both image streams required; run spawn_recording_camera.py first')
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','1280x720','-framerate',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    stop=threading.Event()
    for s in (signal.SIGINT,signal.SIGTERM): signal.signal(s,lambda *_:stop.set())
    started=time.monotonic(); frames=0; dropped=0
    print('RECORDING '+str(a.output),flush=True)
    try:
        with a.output.with_suffix('.frames.jsonl').open('w') as log:
            last=None
            while not stop.is_set() and time.monotonic()-started<a.seconds:
                elapsed=time.monotonic()-started
                if frames/a.fps>elapsed:
                    stop.wait(min(.02,frames/a.fps-elapsed)); continue
                screenshot=n.observer
                canvas=np.zeros((720,1280,3),np.uint8)
                canvas[90:690,:950]=fit(screenshot,950,600)
                if n.rgb is not None: canvas[120:360,960:1280]=fit(n.rgb,320,240)
                def label(text,x,y,size=.55,color=(230,230,230)):
                    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
                label(a.label,20,28,.65)
                label(f'Uncut | 1x wall-clock playback | elapsed {elapsed:.1f}s | sim {n.sim_time}',20,57)
                label('Gazebo observer (recording only)',20,83)
                label('Onboard input RGB',966,109)
                label('Task:',966,395)
                for i,part in enumerate(re.findall('.{1,29}',n.state.get('task','waiting'))):label(part,966,417+i*20,.43)
                label('Safety: '+n.state.get('safety','waiting'),966,505,.45)
                for i,part in enumerate(re.findall('.{1,35}',n.state.get('memory','waiting'))):label(part,966,540+i*18,.4)
                age=None if n.rgb_stamp is None or n.sim_time is None else n.sim_time-n.rgb_stamp
                label('RGB age (sim): '+('unknown' if age is None else f'{age:.3f}s'),966,652,.43)
                encoder.stdin.write(canvas.tobytes()); frames+=1; last=canvas
                log.write(json.dumps(dict(frame=frames-1,wall_elapsed=elapsed,sim_time=n.sim_time,rgb_stamp=n.rgb_stamp,observer_stamp=n.observer_stamp,stream_delta_sec=n.observer_stamp-n.rgb_stamp,rgb_age_sim_sec=age,**n.state))+'\n')
                # Duplicate missing display frames, never speed up the trial.
                while frames/a.fps<time.monotonic()-started-1/a.fps:
                    encoder.stdin.write(last.tobytes()); frames+=1; dropped+=1
                    log.write(json.dumps(dict(frame=frames-1,duplicate=True,wall_elapsed=time.monotonic()-started))+'\n')
    finally:
        encoder.stdin.close(); encoder.wait(timeout=30)
        rclpy.shutdown(); thread.join(timeout=2)
    metadata=dict(output=str(a.output),wall_duration_sec=time.monotonic()-started,frames=frames,fps=a.fps,duplicate_frames=dropped,image_topic=a.image_topic,label=a.label,observer_topic=a.observer_topic,encoder_exit=encoder.returncode,playback='1x wall time; no cuts',synchronization='latest received observer and onboard images; both acquisition stamps logged')
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
    return encoder.returncode

if __name__=='__main__':raise SystemExit(main())
