# OpenArm 實機接入與首測

本手冊供現場交接後使用。本次人手侵入與高抬臂的量測來自 Isaac Sim。移轉時，NUC 的 D415 與 CAN 硬體介面接手模擬資料源，Thor 的 MASt3R 和本機的 MediaPipe、RViz、MoveIt、短軌跡閘控使用相同的程式。[三台主機系統圖](../../docs/figures/chest-curl-architecture.svg)

## 在三台主機建置

每台都在同一版 repo 的根目錄操作。實機使用 domain 88。容器以 Fast DDS、host network 通訊；實驗室既有映像作為建置基底，新容器各自保存程序與相依套件。

| 主機 | 首次建置 | 部署資源 |
|---|---|---|
| 控制 NUC | `bash hardware/right_curl/build.sh robot` | `openarm-dev:latest`、現場 OpenArm 描述、`openarm_can`、`realsense-ros`。可分別設定 `OPENARM_DESCRIPTION_DIR`、`OPENARM_CAN_DIR`、`OPENARM_REALSENSE_DIR` 或共同的 `OPENARM_LAB_SRC` |
| Thor | `bash hardware/guarded_curl/thor.sh build` | 使用實驗室 MASt3R 基底和 adapter；設定同 [Isaac 建置手冊](../isaac_curl/README.md) |
| 本機 | `bash hardware/right_curl/fetch_model.sh`、建置 viewer、`bash hardware/guarded_curl/station.sh build` | 與模擬使用同一套 v1.0 模型和 MoveIt 配置。viewer 建置指令見 [Isaac 建置手冊](../isaac_curl/README.md) |

在 NUC 確認實機目前已交接且 CAN 驅動空閒後，啟動 gateway。它讀取兩臂各 7 個關節與夾爪，發布 TF、關節和一顆胸口 D415。CAN-FD 右臂為 `can0`、左臂為 `can1`。D415 使用獨立容器，僅映射該裝置的 USB、video/media 節點。首次啟動時右肘軌跡控制器為 inactive。遇到已在工作的 gateway 或主動控制器時，先查清持有者與支撐狀態。每次換姿態檔之前，先讓手臂穩定地由現場支撐。

```bash
# 控制 NUC
bash hardware/guarded_curl/gateway.sh

# Thor
bash hardware/guarded_curl/thor.sh up

# 本機，啟動觀測、規劃與 RViz
bash hardware/guarded_curl/station.sh up
bash hardware/guarded_curl/station.sh run observe --chest-only --seconds 10
bash hardware/guarded_curl/station.sh run preflight
```

在工作站容器內啟動 `ros2` CLI 時，它已帶入正確 domain。若自行開終端，環境變數須使用 `export ROS_DOMAIN_ID=88` 或單一指令前綴。確認資料抵達可看 `observe` 的配對影格數、實際頻率及 `preflight` 的逐項結果。`ros2 topic list` 只回報已發現的名稱。

## 確認實體胸口外參與工作姿態

RViz 的預覽可使用既有外參；運動許可使用當次現場驗證的外參。選用固定、可獨立量出基座座標的參考點，分布在桌面工作區。至少四個點估計變換，另三個點留作獨立驗證，最大三維誤差需在 15 mm 內。相機更動後重新採集。

```bash
bash hardware/guarded_curl/station.sh run registration capture /results/calibration/desk
# 把量測與圖上像素填入 host 上 outputs/openarm_hardware/calibration/survey.json
bash hardware/guarded_curl/station.sh run registration solve /results/calibration/survey.json
bash hardware/guarded_curl/station.sh run registration capture-pose
```

`desk.jpg` 可用來讀取像素，`desk.npz` 保存同步深度與內參。`survey.json` 的 `landmarks` 陣列中，每筆須有以下欄位。四個 `fit` 和三個 `holdout` 使用不同標記。下表為格式，數值由現場量測提供。

| 欄位 | 內容 |
|---|---|
| `id`、`reference` | 唯一的點名及其實體定位參考 |
| `role`、`capture` | `fit` 或 `holdout`，以及相對於 manifest 的 NPZ，例如 `desk.npz` |
| `pixel`、`base_m` | 彩色影像整數 `[u,v]` 與 `openarm_body_link0` 座標公尺 `[x,y,z]` |

`capture-pose` 只讀取靜止的真實關節回授，寫入 `tabletop-task.json`。配準與姿態在節點啟動時讀取，完成校正後、控制器仍 inactive 時重啟**本機觀測站**，再執行被動檢查。

```bash
docker restart laiting-openarm-station
bash hardware/guarded_curl/station.sh run preflight
```

### 讓抬肩起始姿態在兩端一致

預設七軸都須在 ±0.16 rad 的放鬆姿態啟動。若胸口看不到下垂夾爪，可在機械支撐下將右臂置於已量測的桌面姿態，再以 `capture-pose` 記錄。桌面模式限定 J1 在 −0.16 至 +0.60 rad、其餘六軸在 ±0.16 rad，且 J4 不低於 −0.02 rad。NUC 啟用時逐軸核對該姿態，最大容許差 0.04 rad。姿態檔同時綁定當次實體 D415 配準檔的 SHA-256。

本機將量測檔送到 NUC 的實驗目錄。下列指令供未來實機交接時使用。若 gateway 仍在支撐手臂，先由現場接手並完成該次實驗，再以新版映像和相同姿態檔重建專用 gateway；腳本遇到既有不同設定會退出。

```bash
# 本機，完成實體外參校正與 capture-pose 之後
ssh tidybot_openarm 'mkdir -p ~/laiting/social-safety-decision/outputs/openarm_hardware/calibration'
scp outputs/openarm_hardware/calibration/tabletop-task.json \
  tidybot_openarm:laiting/social-safety-decision/outputs/openarm_hardware/calibration/

# NUC，在 repo 根目錄；新版映像建置完成、舊實驗已交接
OPENARM_TABLETOP_PROFILE="$PWD/outputs/openarm_hardware/calibration/tabletop-task.json" \
  bash hardware/guarded_curl/gateway.sh
```

工作站讀取 gateway 透過 `/robot_description` 公布的姿態檔摘要。抬肩模式要求兩端摘要一致、實際關節與目標一致，且胸口仍看得到完整夾爪路徑。實機首測程式請求 18°，CAN 的肘角上限仍為起始角加 0.46 rad，速度上限為 0.14 rad/s。模擬的 36° 高彎舉另需實體行程與制動驗收。若視野檢查不通過，依胸口影像修正實際安裝或起始姿態，再重新校正與記錄。

## 現場低速驗收

先讓手臂保持靜止，以可辨識的手模型進出桌面區域，核對 `/laiting/safety/state` 的保持與淨空轉換。右手空載、可操作硬體急停且 `preflight` 全部通過時，由工作站先執行規劃，再執行單回合。軌跡只由一個發送端持有。

```bash
bash hardware/guarded_curl/station.sh run repeat
bash hardware/guarded_curl/station.sh run repeat --execute --cycles 1 --timeout 120
```

保持動作會鎖住量測到的肘角；失去 RGB-D、點雲或許可心跳也會停止續送動作。垂手模式在回到起始位置且速度足夠低後停用控制器。抬肩模式即使完成回合也保持支撐。硬體停用會解除支撐，若出現 `support_retained`，應由現場操作員接手支撐。實機首測須以實際回授、實際止動距離與急停功能重新驗收。觀測與 CSV 存放於工作站 `outputs/openarm_hardware/`。
