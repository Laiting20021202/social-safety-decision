#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
world=laiting-isaac-fixed-world
station=laiting-isaac-curl-station
out="$here/../../outputs/isaac_guarded"
mkdir -p "$out"
exec 9>"$out/demo.lock"
flock -n 9 || { echo "An Isaac recording is already running" >&2; exit 1; }
bash "$here/world.sh" >/dev/null
start=$(docker inspect -f '{{.State.StartedAt}}' "$world")
for ((i=0; i<240; ++i)); do
  if docker logs --since "$start" "$world" 2>&1 | awk '/Isaac ready/ {found=1} END {exit !found}'; then break; fi
  sleep 1
done
if (( i==240 )); then echo 'Isaac did not publish its stage and encoders' >&2; exit 1; fi
bash "$here/control.sh" up >/dev/null
for ((i=0; i<40; ++i)); do
  window=$(xwininfo -root -tree | awk '/chest.rviz.*RViz/ {if(!w) w=$1} END {print w}')
  if [[ -n $window ]]; then break; fi
  sleep 1
done
[[ -n $window ]] || { echo 'The chest.rviz window is unavailable on this display' >&2; exit 1; }
station_exec=(docker exec -i "$station" bash /opt/right-curl/app/env.sh python3)
font=${DEMO_CJK_FONT:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}
[[ -r $font ]] || { echo "Set DEMO_CJK_FONT to NotoSansCJK-Regular.ttc for the video captions" >&2; exit 1; }
docker cp "$font" "$station:/tmp/laiting-chinese.ttc"
"${station_exec[@]}" /demo/isaac/record_trial.py --window "$((window))" "$@"
"${station_exec[@]}" /demo/isaac/validate.py
"${station_exec[@]}" /demo/isaac/demo_video.py
"${station_exec[@]}" /demo/isaac/demo_video.py --realtime
