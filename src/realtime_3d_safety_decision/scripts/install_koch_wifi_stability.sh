#!/usr/bin/env bash
# Install on the Koch NUC (192.168.0.231), not on the 3D workstation.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo --preserve-env=KOCH_LAN_PEER bash "$0" "$@"
fi

PEER_IP="${KOCH_LAN_PEER:-192.168.0.234}"
LAN_INTERFACE="$(ip route get "${PEER_IP}" 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -n1)"
LAN_INTERFACE="${LAN_INTERFACE:-wlx08beac4c6d30}"

# NetworkManager value 2 means "disable Wi-Fi power saving". Persist the
# connection setting and apply it without deliberately cycling the connection.
CONNECTION="$(nmcli -g GENERAL.CONNECTION device show "${LAN_INTERFACE}" 2>/dev/null || true)"
if [[ -n "${CONNECTION}" && "${CONNECTION}" != -- ]]; then
  nmcli connection modify "${CONNECTION}" 802-11-wireless.powersave 2
  nmcli device reapply "${LAN_INTERFACE}" >/dev/null 2>&1 || true
fi

# Both Koch computers use an Edimax 7392:f822 / Realtek rtw88 USB adapter.
# Prevent USB autosuspend and rtw88 deep low-power mode from silently dropping
# the long-running MJPEG/DDS link.
install -Dm644 /dev/stdin /etc/udev/rules.d/80-koch-wifi-power.rules <<'EOF'
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="7392", ATTR{idProduct}=="f822", TEST=="power/control", ATTR{power/control}="on"
EOF
install -Dm644 /dev/stdin /etc/modprobe.d/rtw88-koch-stability.conf <<'EOF'
options rtw88_core disable_lps_deep=y
EOF

install -Dm755 /dev/stdin /usr/local/sbin/koch-wifi-stability <<EOF
#!/usr/bin/env bash
set -euo pipefail
peer='${PEER_IP}'
interface=\$(ip route get "\${peer}" 2>/dev/null | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p' | head -n1)
interface=\${interface:-${LAN_INTERFACE}}
connection=\$(nmcli -g GENERAL.CONNECTION device show "\${interface}" 2>/dev/null || true)
if [[ -n "\${connection}" && "\${connection}" != -- ]]; then
  nmcli connection modify "\${connection}" 802-11-wireless.powersave 2
  nmcli device reapply "\${interface}" >/dev/null 2>&1 || true
fi
if [[ -w /sys/module/rtw88_core/parameters/disable_lps_deep ]]; then
  printf 'Y\\n' >/sys/module/rtw88_core/parameters/disable_lps_deep
fi
for device in /sys/bus/usb/devices/*; do
  [[ -f "\${device}/idVendor" && -f "\${device}/idProduct" ]] || continue
  if [[ "\$(<"\${device}/idVendor")" == 7392 && "\$(<"\${device}/idProduct")" == f822 ]]; then
    printf 'on\\n' >"\${device}/power/control" 2>/dev/null || true
  fi
done
EOF

install -Dm644 /dev/stdin /etc/systemd/system/koch-wifi-stability.service <<'EOF'
[Unit]
Description=Keep the Koch USB Wi-Fi link out of power-saving states
Wants=network-online.target
After=network-online.target NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/koch-wifi-stability
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

udevadm control --reload-rules
systemctl daemon-reload
systemctl enable --now koch-wifi-stability.service

echo "interface=${LAN_INTERFACE}"
echo "connection=${CONNECTION:-unknown}"
echo "NetworkManager powersave=$(nmcli -g 802-11-wireless.powersave connection show "${CONNECTION}" 2>/dev/null || echo unknown)"
if [[ -r /sys/module/rtw88_core/parameters/disable_lps_deep ]]; then
  echo "rtw88 disable_lps_deep=$(</sys/module/rtw88_core/parameters/disable_lps_deep)"
fi
echo "Installed and enabled koch-wifi-stability.service."
