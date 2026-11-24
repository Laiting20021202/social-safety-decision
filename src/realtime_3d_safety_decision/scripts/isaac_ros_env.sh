#!/usr/bin/env bash
# Source from a Bash terminal. Discover Isaac Sim through LAN multicast.
_isaac_ros_env() {
  local lan_ip profile_dir profile restore_nounset=0 setup_status=0
  lan_ip="$(ip -4 route get 239.255.0.1 | sed -n 's/.* src \([^ ]*\).*/\1/p' | head -n1)"
  if [[ -z "$lan_ip" || "$lan_ip" == 127.* ]]; then
    echo 'No LAN IPv4 interface for DDS multicast.' >&2
    return 1
  fi
  profile_dir="${XDG_CONFIG_HOME:-$HOME/.config}/isaac-ros-lan"
  mkdir -p "$profile_dir" || return 1
  profile="$profile_dir/fastdds-${lan_ip}.xml"
  cat > "$profile" <<XML
<?xml version="1.0" encoding="UTF-8"?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors><transport_descriptor>
    <transport_id>isaac_udp</transport_id><type>UDPv4</type>
    <sendBufferSize>1048576</sendBufferSize><receiveBufferSize>8388608</receiveBufferSize>
    <interfaceWhiteList><address>${lan_ip}</address></interfaceWhiteList>
  </transport_descriptor></transport_descriptors>
  <participant profile_name="isaac_lan" is_default_profile="true"><rtps>
    <userTransports><transport_id>isaac_udp</transport_id></userTransports>
    <useBuiltinTransports>false</useBuiltinTransports>
    <builtin>
      <metatrafficUnicastLocatorList><locator><udpv4><address>${lan_ip}</address><port>0</port></udpv4></locator></metatrafficUnicastLocatorList>
      <metatrafficMulticastLocatorList><locator><udpv4><address>239.255.0.1</address><port>7400</port></udpv4></locator></metatrafficMulticastLocatorList>
      <initialPeersList><locator><udpv4><address>239.255.0.1</address><port>7400</port></udpv4></locator></initialPeersList>
    </builtin>
  </rtps></participant>
</profiles>
XML
  [[ $- == *u* ]] && restore_nounset=1
  set +u
  source /opt/ros/humble/setup.bash || setup_status=$?
  if [[ "$restore_nounset" == 1 ]]; then set -u; fi
  if [[ "$setup_status" != 0 ]]; then return "$setup_status"; fi
  unset CYCLONEDDS_URI ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT ROS_LAN_PEER
  unset FASTDDS_DEFAULT_PROFILES_FILE FASTRTPS_DEFAULT_PROFILES_FILE
  unset RMW_FASTRTPS_USE_QOS_FROM_XML FASTDDS_BUILTIN_TRANSPORTS
  export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTRTPS_DEFAULT_PROFILES_FILE="$profile"
  echo "Isaac ROS: ${lan_ip}, domain 0, Fast DDS UDP, multicast discovery"
}
if _isaac_ros_env; then
  unset -f _isaac_ros_env
else
  unset -f _isaac_ros_env
  return 1 2>/dev/null || exit 1
fi
