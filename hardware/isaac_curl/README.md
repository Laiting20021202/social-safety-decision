# 重跑固定 OpenArm 的胸口 RGB-D 實驗

本機需要支援 Isaac Sim 4.5 的 NVIDIA GPU、Docker GPU runtime、X11 桌面、`xwininfo` 與 `flock`。Thor 與本機位於同一個可傳遞 DDS 的區網，主機時鐘需同步。ROS 節點、深度推論、MoveIt、RViz、Isaac 和影片編碼都在各自的 Docker 內執行。主機腳本負責 Docker、顯示授權與檔案。錄影使用 Noto Sans CJK TC；可透過 `DEMO_CJK_FONT` 指定 `NotoSansCJK-Regular.ttc` 的位置。

## 首次建置

在本機 repo 根目錄執行。若已有對應的映像，直接使用下節的重跑指令。

```bash
bash docker/build.sh
bash hardware/right_curl/fetch_model.sh
OPENARM_DESCRIPTION_DIR="$PWD/.openarm-vendor/nuc-v10-description" \
  BASE_IMAGE=vx300s-ros2:humble bash hardware/right_curl/build.sh viewer
bash hardware/guarded_curl/station.sh build
bash hardware/isaac_curl/control.sh build
```

`vx300s-ros2:humble` 是本次工作站已有的基底。其他工作站可指定 `BASE_IMAGE=osrf/ros:humble-desktop`。模擬使用現場 v1.0 的 URDF、夾爪與碰撞網格。模型腳本由固定的上游 commit 套用現場差異，依 SHA-256 清單驗證。LibHand 網格和紋理已附在 repo。[模型差異](../right_curl/site-v10.patch) · [模型清單](../right_curl/site-v10.sha256) · [人手模型歸屬](assets/libhand/ATTRIBUTION.md)

在 Thor 的同一版本 repo 執行。基底 `pure-mast3r-d455-unity:self-contained` 包含實驗室已有的 MASt3R-SLAM 程式、CUDA 相依與模型權重；建置時從本機目錄複製 `thor_live.py`、`thor_geometry.py`、`thor_tracker.py` 到新映像。這三個 adapter 的預設目錄為 `~/mast3r_openvins/master_openarm`，可透過 `THOR_MAST3R_ADAPTER` 指定。此實驗使用固定相機逐幀推論。

```bash
bash hardware/isaac_curl/thor.sh build
bash hardware/isaac_curl/thor.sh up
```

## 錄製與檢查

在本機帶有 `DISPLAY` 的終端執行。接受 NVIDIA Isaac Sim 條款後設定 `ACCEPT_EULA=Y`。Thor 容器需先啟動。

```bash
ACCEPT_EULA=Y bash hardware/isaac_curl/demo.sh --cycles 2 --timeout 220
```

`--cycles` 可設為大於 2 的整數。每次試驗有一次突然伸手，發生於第一趟上抬。調整回合數時應預留對應的逾時時間。腳本等待有效許可才啟動關節控制器。動態人手資料僅供場景和驗證器使用。錄製時保持胸口 RViz 視窗可見。腳本在 `outputs/isaac_guarded/` 更新當次影片、時間戳和量測，已發布的場次另存在 `results/`。

| 輸出 | 內容 |
|---|---|
| `openarm-chest-demo.mp4`、`openarm-chest-realtime.mp4` | 同步胸口 RGB 與真實 RViz 視窗。示範版在人手伸入、停留與恢復時使用 1 倍速，其餘 4 倍；原速版全程 1 倍 |
| `commands.csv`、`physics.csv` | 動作命令、保持與重規劃時刻，以及每物理步的人手姿態、實際關節與辨識狀態 |
| `clearance.csv`、`physx-contacts.json`、`validation.json` | 網格最短距離、PhysX 接觸回報、逐項驗證結果 |
| `isaac-raw.avi`、`rviz-timed.mkv`、`frame-times.csv` | 未疊字相機、保留壁鐘時間戳的 RViz 與逐幀對時資料 |

桌面在基座座標的高度為 0.098 m，虛擬胸口相機使用 640 × 480。右肩預置 0.55 rad，右肘從 0.03 rad 起彎舉 36°。影片顯示回授角度。可用下列命令在相同容器內檢查一顆相機的資料流、控制器與點雲。

```bash
bash hardware/isaac_curl/control.sh run observe --chest-only --seconds 10
bash hardware/isaac_curl/control.sh run preflight
bash hardware/isaac_curl/control.sh ros2 topic info -v /laiting/scene/points
bash hardware/isaac_curl/control.sh ros2 control list_controllers
```

啟動後長時間等待時，先看 `/laiting/safety/state` 的 `reason`。若為 `mast3r_unavailable`，檢查 Thor 日誌、模型權重和 domain 89 的跨主機通訊。若為 `tool_path_out_of_view` 或 `tool_not_visible_or_occluded`，檢查胸口影像、TF 和夾爪路徑。試驗停止在抬臂位置時，控制器會保留支撐；模擬可在確認錄影已結束後，停止 **domain 89 的兩個本機容器**，重新啟動場景與控制站回到初始姿態。

```bash
docker stop laiting-isaac-curl-station laiting-isaac-fixed-world
# Thor 另行在其主機結束本次模擬推論
docker stop laiting-isaac-thor
```

模型或控制映像有更動時，結束該次模擬、移除這兩個已停止的本機模擬容器，再用 `demo.sh` 建立。已發布的資料與離線重算請見[本次實驗紀錄](../../results/isaac_chest_curl_2026-11-26/README.md)。
