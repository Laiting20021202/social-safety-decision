#!/usr/bin/env bash
# Source this file in every ROS 2 terminal on the 3D host.
# Domain 42 is shared with the Koch camera computer. Multicast is used for
# LAN-wide participant discovery and the camera is retained as a unicast
# fallback because this USB Wi-Fi/AP path has shown intermittent multicast.

# Gazebo and its perception GUI run on this host. Keep their discovery and
# commands separate from the physical-camera LAN without changing its defaults.
if [[ "${ROS_LOCAL_SIM:-0}" == "1" ]]; then
  export ROS_DOMAIN_ID="${KOCH_ROS_DOMAIN_ID:-79}"
  # Humble's RMW adds lo itself when LOCALHOST_ONLY=1; an explicit lo XML
  # selection then fails with "same interface may not be selected twice".
  # The XML below already restricts traffic to loopback.
  export ROS_LOCALHOST_ONLY=0
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces><AllowMulticast>true</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
  unset ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE FASTRTPS_DEFAULT_PROFILES_FILE
  return 0
fi

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ROS_LAN_PEER="${ROS_LAN_PEER:-192.168.0.231}"
ROS_LAN_INTERFACE="${ROS_LAN_INTERFACE:-$(
  ip route get "${ROS_LAN_PEER}" 2>/dev/null |
    sed -n 's/.* dev \([^ ]*\).*/\1/p' |
    head -n1
)}"
if [[ -z "${ROS_LAN_INTERFACE}" || "${ROS_LAN_INTERFACE}" == "lo" ]]; then
  ROS_LAN_INTERFACE="$(
    ip route show default 2>/dev/null |
      sed -n 's/.* dev \([^ ]*\).*/\1/p' |
      head -n1
  )"
fi
if [[ -n "${ROS_LAN_INTERFACE}" ]]; then
  export CYCLONEDDS_URI="<CycloneDDS xmlns=\"https://cdds.io/config\"><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"${ROS_LAN_INTERFACE}\" multicast=\"true\"/></Interfaces><AllowMulticast>spdp</AllowMulticast><EnableMulticastLoopback>true</EnableMulticastLoopback></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex><SPDPInterval>2s</SPDPInterval><Peers><Peer Address=\"${ROS_LAN_PEER}\"/></Peers></Discovery></Domain></CycloneDDS>"
fi

# Do not let settings for another DDS implementation override Cyclone DDS.
unset ROS_DISCOVERY_SERVER FASTDDS_DEFAULT_PROFILES_FILE FASTRTPS_DEFAULT_PROFILES_FILE
