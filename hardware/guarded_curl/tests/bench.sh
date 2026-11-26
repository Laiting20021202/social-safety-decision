#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
results="$root/outputs/openarm_hardware"
mkdir -p "$results"
mode=${1:-sensor}
[[ "$mode" == sensor || "$mode" == heartbeat ]] || exit 2
if [[ "$mode" == sensor && ! -f "$results/test-inputs/right_hands.jpg" ]]; then
  mkdir -p "$results/test-inputs"
  docker run --rm --entrypoint curl --user "$(id -u):$(id -g)" -v "$results/test-inputs:/inputs" \
    laiting-openarm-safety:humble -fsSL https://storage.googleapis.com/mediapipe-assets/right_hands.jpg -o /inputs/right_hands.jpg
fi
if [[ "$mode" == sensor ]]; then
  echo "4b5134daa4cb60465535239535f9f74c2842aba3aa5fd30bf04ef5678f93d87f  $results/test-inputs/right_hands.jpg" | sha256sum --check --status
fi
docker run --rm --name laiting-curl-isolated-check --network none --init \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_HOME=/tmp/ros -e ROS_DOMAIN_ID=158 -e "BENCH_MODE=$mode" --tmpfs /results:rw,mode=1777 \
  -v "$here:/bench:ro" -v "$here/../right_curl:/mock:ro" -v "$results:/evidence" \
  -v "$root/src/openarm_bimanual_moveit_config/config/openarm_v1.0:/opt/laiting/moveit:ro" \
  laiting-openarm-safety:humble bash -c '
    set -eu
    ros2 launch /mock/right.launch.py mock:=true >/tmp/mock.log 2>&1 & robot=$!
    ros2 launch /bench/station.launch.py planner_only:=true >/tmp/moveit.log 2>&1 & planner=$!
    if [[ "$BENCH_MODE" == sensor ]]; then
      python3 -u /bench/tests/replay_rgbd.py >/tmp/fixture.log 2>&1 & fixture=$!
      for _ in $(seq 1 80); do [[ ! -s /results/calibration/d415.json ]] || break; sleep .1; done
      python3 /bench/guard.py >/tmp/hand.log 2>&1 & hand=$!
    else
      python3 /bench/tests/replay_guard.py >/tmp/fixture.log 2>&1 & fixture=$!
    fi
    stop() { kill -INT "$robot" "$planner" "$fixture" ${hand:+"$hand"} 2>/dev/null || true; }
    trap stop EXIT
    sleep 5
    log=/evidence/isolated-$BENCH_MODE-$(date +%s).csv
    if ! python3 /bench/repeat.py --execute --cycles 2 --log "$log"; then
      tail -35 /tmp/{mock,moveit,fixture}.log; exit 1
    fi
    python3 /bench/tests/check_run.py "$log"
    if [[ "$BENCH_MODE" == sensor ]]; then
      cp /tmp/fixture.log "${log%.csv}.log"
      grep "guard hand_in_workspace" /tmp/fixture.log
      grep "guard no_fresh_rgbd" /tmp/fixture.log
    fi
  '
