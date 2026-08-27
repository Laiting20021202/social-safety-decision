# EdgeTAM + Point-cloud Tracker：離線合成評估

格式版本：3

## 結論

本次 deterministic point-cloud-only functional evaluation：**10/10 scenarios passed**。

這個deterministic評估器本身不是EdgeTAM、相機、ROS graph、CBF或馬達效能
benchmark；它沒有載入checkpoint或執行GPU inference。另有一次真實官方
checkpoint CUDA/API smoke，但repository仍沒有同步RGB-D rosbag或ground
truth；因此live端到端FPS、mask IoU/accuracy、融合accuracy與控制成功率仍
明確列為 **N/A / not measured**，不由合成資料推估。

## 實際執行範圍

- 固定亂數種子與 0.1 s sensor time step；不記錄 wall-clock timing。
- 直接執行 point-cloud cluster quality、6D Kalman/Hungarian association、
  track lifecycle、mask-invalid geometry fallback，以及 moving-camera 到
  tracking-frame 的 rigid transform/preprocess/cluster path。
- EdgeTAM mask drift案例只測 quality gate與 point-cloud fallback；不是 model
  accuracy案例。
- 完整機器可讀資料：`results/edgetam_pointcloud_evaluation.csv`。

## Requested mode/metric comparison

沒有共同RGB-D ground truth可讓四種模式公平比較；未執行欄位維持N/A。
point-cloud-only欄只使用本評估器實際產生的deterministic synthetic結果，
不能當成真實sensor accuracy。ROS point-cloud snapshot與官方EdgeTAM API smoke
分別記錄於`results/ros_live_smoke_2026-07-31.md`與
`results/edgetam_official_smoke.md`，都不冒充同步RGB-D平均benchmark。

| Metric | Original pipeline | Point-cloud-only | EdgeTAM-only | EdgeTAM + cloud |
|---|---|---|---|---|
| detected obstacle count | N/A (different video pipeline) | 10/10 expected scenario outcomes passed | N/A (not run by this evaluator) | N/A (not run) |
| missed obstacle count | N/A | 0 scenario-level outcome failures; per-frame count N/A | N/A | N/A |
| false obstacle count | N/A | N/A (no labeled background-only sequence) | N/A | N/A |
| ID switch count | N/A | 0 unintended; 1 expected after expiry/re-entry | N/A | N/A |
| average track duration | N/A | N/A (scenario lengths intentionally differ) | N/A | N/A |
| position jitter | N/A | N/A; per-scenario mean position error is reported below | N/A | N/A |
| velocity jitter | N/A | N/A (only final velocity accuracy asserted) | N/A | N/A |
| mask-cluster IoU | N/A | N/A; drift case intentionally has no overlap | N/A | N/A |
| average FPS | N/A | N/A (wall-clock timing intentionally excluded) | N/A（另見獨立API smoke報告） | N/A |
| average latency | N/A | N/A | N/A（另見獨立API smoke報告） | N/A |
| maximum latency | N/A | N/A | N/A（另見獨立API smoke報告） | N/A |
| nearest-distance stability | N/A | N/A (surface validity asserted, no noise GT series) | N/A | N/A |

## Scenario results

| Scenario | Result | ID switches | Mean position error (m) | Final state | Point-cloud outcome |
|---|---:|---:|---:|---|---|
| static | PASS | 0 | 0.000612 | CONFIRMED | GOOD geometry retained |
| moving | PASS | 0 | 0.001237 | CONFIRMED | GOOD geometry retained |
| crossing | PASS | 0 | 0.001994 | CONFIRMED+CONFIRMED | Two point-cloud tracks |
| occlusion | PASS | 0 | 0.000972 | CONFIRMED | Prediction during 3 missing frames |
| leave_reappear | PASS | 1 | 0.000000 | CONFIRMED | Expired geometry not held indefinitely |
| sparse | PASS | 0 | 0.000000 | CONFIRMED | SPARSE |
| mask_drift | PASS | 0 | 0.000000 | CONFIRMED | fallback:mask_quality_invalid |
| rgb_missing | PASS | 0 | 0.000988 | CONFIRMED | GOOD geometry; mask UNAVAILABLE |
| robot_motion | PASS | 0 | 0.000000 | CONFIRMED | TF-transformed point-cloud track |
| hand_entrance | PASS | 0 | 0.000000 | CONFIRMED | Emergency near-field point-cloud track |

## Pass criteria與案例意義

- `static`、`moving`：同一物體維持一個 ID，進入 CONFIRMED；moving另檢查
  最終速度誤差。
- `crossing`：兩個尺寸/點數不同但路徑交會的 cluster各自維持唯一 ID。
- `occlusion`：三個 frame缺點雲時只進入 OCCLUDED/LOST，重新出現仍是
  原 ID。
- `leave_reappear`：超過 retention後刪除舊 track，稍後回來必須配置新 ID；
  此案例的 `ID switches=1` 是預期行為。
- `sparse`：有限但稀疏的點雲標成 SPARSE，不被誤當成無障礙物。
- `mask_drift`：無重疊的 synthetic mask被標成 DEGRADED/INVALID，fusion
  保留非空 point-cloud fallback。
- `rgb_missing`：整段沒有 RGB/mask，點雲 ID仍連續且 mask quality保持
  UNAVAILABLE。
- `robot_motion`：把移動 camera frame的點先轉入固定 tracking frame，再
  經 preprocess、Euclidean cluster與 tracker；固定物體不得產生 ID switch。
- `hand_entrance`：近於 emergency distance的新 cluster不等待一般
  confirmation hits，第一個 measurement立即 CONFIRMED。

## 尚未驗證（N/A）

| 項目 | 結果 | 需要的真實輸入 |
|---|---|---|
| EdgeTAM load / propagation | 不屬於本評估器；另見 `results/edgetam_official_smoke.md` | 官方checkpoint + CUDA/bf16 + 2 IDs + ID刪除 |
| EdgeTAM mask accuracy | N/A | 有標註RGB-D ground truth masks |
| RGB-D mask/cloud fusion accuracy | N/A | 同步 RGB、metric depth/organized cloud、CameraInfo與 GT |
| Live ROS sync、TF drop、sensor stale recovery | N/A | 可 replay 的 rosbag與完整 TF tree |
| 離線EdgeTAM API latency / VRAM | 不屬於本評估器；另見 `results/edgetam_official_smoke.md` | 只代表該次獨立smoke |
| ROS end-to-end FPS / p50 / p95 latency / VRAM | N/A | 目標RTX 4060 Ti上的同步RGB-D live run |
| Robot self-filter recall | N/A | URDF/link frames與 robot/non-robot point labels |
| CBF / motor safety behavior | N/A | 外部 controller repository與硬體測試程序 |

## 重現

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python tools/evaluate_tracker.py
```

只要任何 scenario不符合 pass criteria，程式會以 non-zero status結束；
CSV與Markdown仍會先寫出，以便診斷。
