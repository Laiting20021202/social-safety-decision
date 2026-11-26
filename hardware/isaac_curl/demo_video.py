"""Compose synchronized chest RGB, actual RViz and measured mesh clearances into an H.264 demo."""
import argparse
import csv
from bisect import bisect_right
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 15
BG = (20, 29, 38)
WHITE = (238, 244, 246)
MUTED = (169, 187, 193)
GREEN = (106, 224, 171)
AMBER = (255, 192, 107)


def read_rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


@lru_cache()
def font(size):
    return ImageFont.truetype(os.environ.get('DEMO_CJK_FONT','/tmp/laiting-chinese.ttc'), size, index=3)


def label(draw, x, y, text, size, color=WHITE):
    draw.text((x, y), text, font=font(size), fill=color, anchor='lt')


def relative_index(stamps, at):
    return max(0, bisect_right(stamps, at)-1)


class Video:
    def __init__(self, path, times):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened() or not len(times):
            raise RuntimeError('unreadable video '+str(path))
        self.times = times
        self.i = -1
        self.frame = None

    def at(self, t):
        index = relative_index(self.times, t)
        while self.i < index:
            ok, self.frame = self.capture.read()
            if not ok:
                raise RuntimeError('source video ended before its timestamp list')
            self.i += 1
        return self.frame


def viewport(raw):
    # RViz's Qt main window contains the panel toolbar and the 3-D viewport.
    h, w = raw.shape[:2]
    view = raw[round(h*.085):round(h*.955),round(w*.333):round(w*.978)]
    scale = min(590/view.shape[1], 480/view.shape[0])
    return cv2.cvtColor(cv2.resize(view,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('/results'))
    parser.add_argument('--realtime', action='store_true', help='Keep the entire trial at original wall-clock speed')
    args = parser.parse_args()
    folder = args.folder
    verified = json.loads((folder/'validation.json').read_text())
    if not verified['passed']:
        raise SystemExit('verified trial required for presentation')
    phys = read_rows(folder/'physics.csv')
    times = [int(r['wall_ns']) for r in phys]
    commands = read_rows(folder/'commands.csv')
    clearances = read_rows(folder/'clearance.csv')
    if len(clearances) != len(times) or any(int(r['wall_ns'])!=t for r,t in zip(clearances,times)):
        raise ValueError('clearance file and physics recording belong to different trials')
    frame_ns = [int(r['wall_ns']) for r in read_rows(folder/'frame-times.csv')]
    pts = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
          '-show_entries','packet=pts_time','-of','csv=p=0',str(folder/'rviz-timed.mkv')], text=True)
    rviz_ns = [round(float(t)*1e9) for t in pts.splitlines() if t]
    if abs(frame_ns[0]-times[0])>200_000_000 or rviz_ns[0]>frame_ns[0] or frame_ns[-1]>rviz_ns[-1]+200_000_000:
        raise ValueError('unsynchronized camera, RViz or physics recording')
    chest = Video(folder/'isaac-raw.avi',frame_ns)
    rviz = Video(folder/'rviz-timed.mkv',rviz_ns)
    actions = [r for r in commands if r['event'] in ('stream','hold','cycle','released')]
    action_times = [int(r['wall_ns']) for r in actions]
    clear = np.array([float(r['clearance_m'])*100 for r in clearances])
    elbow = np.array([np.rad2deg(float(r['openarm_right_joint4'])) for r in phys])
    first_hold = next(int(r['wall_ns']) for r in actions if r['event']=='hold' and r['reason']=='hand_in_workspace')
    resume = next(int(r['wall_ns']) for r in actions if r['event']=='stream' and int(r['wall_ns'])>first_hold)
    entered = next(t for r,t in zip(phys,times) if r['phase']=='entering')
    withdrawal = next(t for r,t in zip(phys,times) if r['phase']=='withdrawn')
    duration = (frame_ns[-1]-frame_ns[0])*1e-9
    base = Image.new('RGB',(1280,736),BG)
    d = ImageDraw.Draw(base)
    label(d,18,14,'OpenArm　右臂彎舉與人手避碰',32)
    label(d,1031,26,'Isaac Sim 4.5  /  ROS 2',18,MUTED)
    label(d,20,72,'胸口 D415　彩色影像',20)
    label(d,674,72,'RViz　關節回授與 Thor 融合點雲',20)
    d.rectangle((670,103,1261,583),fill=(43,47,53))
    label(d,20,667,'右肘角度',17,MUTED)
    label(d,250,667,'人手至機器人網格最短距離',17,MUTED)
    label(d,620,667,'實驗經過時間',17,MUTED)
    label(d,875,667,'右肘角度隨時間變化',17,MUTED)
    x0, x1, y0, y1 = 875,1251,694,720
    gx = x0+(np.array(times,dtype=float)-times[0])/(times[-1]-times[0])*(x1-x0)
    gy = y1-(elbow-min(elbow))/(max(elbow)-min(elbow))*(y1-y0)
    d.line([(float(x),float(y)) for x,y in zip(gx[::25],gy[::25])],fill=(85,112,122),width=2)
    target=folder/('openarm-chest-realtime.mp4' if args.realtime else 'openarm-chest-demo.mp4')
    pending=target.with_suffix('.partial.mp4')
    encode=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x736','-r',str(FPS),'-i','-','-c:v','libx264','-threads','4','-preset','fast',
        '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',
        '-metadata','comment=Hand model: Marin Šarić, LibHand 0.9 (2011), CC BY 3.0, https://libhand.org/. Adapted spread-hand pose and sleeve.',
        str(pending)],stdin=subprocess.PIPE)
    t=frame_ns[0]
    saved=False
    try:
        while t<=frame_ns[-1]:
            i=relative_index(times,t)
            row=phys[i]
            action=actions[relative_index(action_times,t)] if t>=action_times[0] else {}
            image=base.copy()
            image.paste(Image.fromarray(cv2.cvtColor(chest.at(t),cv2.COLOR_BGR2RGB)),(18,104))
            view=viewport(rviz.at(t))
            image.paste(Image.fromarray(view),(670+(592-view.shape[1])//2,104+(480-view.shape[0])//2))
            d=ImageDraw.Draw(image)
            if first_hold<=t<resume:
                color=AMBER
                if row['guard']=='hand_in_workspace':
                    message='辨識到人手　右臂保持位置'
                elif t<withdrawal:
                    message='人手撤離中　右臂保持位置'
                else:
                    message='人手已撤離　等待工作區持續淨空'
            elif row['phase']=='entering':
                color=AMBER
                message='人手正伸入桌面工作區'
            elif t>=resume and t<resume+2.5e9:
                color=GREEN
                message='MoveIt 從目前關節位置重新規劃，繼續抬手'
            elif action.get('event')=='stream':
                color=GREEN
                message='右手抬起' if float(action['target_rad'])>.3 else '右手放下'
            elif action.get('event') in ('cycle','released'):
                color=GREEN;message='完成抬放，返回起始位置'
            else:
                color=MUTED;message='等待回授與運動規劃'
            d.rectangle((18,602,23,639),fill=color)
            label(d,36,605,message,26,color)
            speed=1 if args.realtime or entered-2e9<=t<=resume+2.5e9 else 4
            label(d,1098,611,f'播放速度  {speed}×',19,MUTED)
            label(d,20,694,f'{elbow[i]:.1f}°',26)
            label(d,250,694,f'{clear[i]:.1f} cm',26,AMBER if first_hold<=t<resume else WHITE)
            label(d,620,694,f'{(t-frame_ns[0])*1e-9:05.1f} / {duration:.1f} s',24)
            dot=(float(gx[i]),float(gy[i]))
            d.ellipse((dot[0]-3,dot[1]-3,dot[0]+3,dot[1]+3),fill=color)
            encode.stdin.write(image.tobytes())
            if not saved and row['phase']=='present' and row['guard']=='hand_in_workspace' and t>=first_hold+1.1e9:
                if not args.realtime:
                    image.save(folder/'demo-hold.jpg',quality=93)
                saved=True
            t+=round(1e9*speed/FPS)
    finally:
        encode.stdin.close()
        chest.capture.release()
        rviz.capture.release()
        code=encode.wait()
    if code or not saved:
        raise SystemExit('could not produce verified video and hand frame')
    pending.replace(target)
    print(target,flush=True)


if __name__=='__main__':
    main()
