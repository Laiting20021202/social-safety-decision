"""Record the actual RViz window during the camera-driven Isaac trial."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=220.)
    args = parser.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') != '89':
        raise SystemExit('recording requires the Isaac station')
    dest = Path('/results/rviz-timed.mkv')
    with open('/tmp/chest-rviz-encoder.log', 'w') as log:
        record = subprocess.Popen([
            'ffmpeg', '-nostdin', '-loglevel', 'error', '-copyts', '-use_wallclock_as_timestamps', '1',
            '-f', 'x11grab', '-framerate', '8', '-draw_mouse', '0', '-window_id', str(args.window),
            '-i', os.environ['DISPLAY'], '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p', '-bf', '0', '-vsync', '0', '-avoid_negative_ts', 'disabled',
            '-y', str(dest)
        ], stderr=log)
        try:
            time.sleep(.8)
            if record.poll() is not None:
                raise RuntimeError(Path(log.name).read_text())
            subprocess.run([sys.executable, '/demo/isaac/trial.py', '--cycles', str(args.cycles),
                            '--timeout', str(args.timeout)], check=True)
        finally:
            if record.poll() is None:
                record.send_signal(signal.SIGINT)
                record.wait(timeout=12)


if __name__ == '__main__':
    main()
