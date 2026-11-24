# 歷史交付索引

| 階段 | 可核對的版本與交付 |
|---|---|
| 2D 感知與 GUI | [2d_version](https://github.com/Laiting20021202/social-safety-decision/tree/2d_version) 保留早期 2D 偵測、可視化與安全判斷。 |
| 3D 感知與 ROS 2 | [3d_version](https://github.com/Laiting20021202/social-safety-decision/tree/3d_version)、[感知層](https://github.com/Laiting20021202/social-safety-decision/tree/layer/perception-detection)、[早期 ROS smoke](../results/ros_live_smoke_2026-07-31.md)、[EdgeTAM 評估](../results/edgetam_pointcloud_evaluation.md)。 |
| OpenArm 與動態避障 | [控制層](https://github.com/Laiting20021202/social-safety-decision/tree/layer/dynamic-avoidance-control) 保留 Safety Bridge、MoveIt 和動態控制，驗收紀錄見[模擬交接](../src/openarm_sim/docs/HANDOFF_STATUS.md)。 |
| Gazebo／Isaac 模擬 | [模擬層](https://github.com/Laiting20021202/social-safety-decision/tree/layer/simulation-validation)、[複合對照 JSON](../results/gazebo_restore/)、[驗收協定](../src/openarm_sim/docs/validation_protocol.md)。 |
| 2026-09-22 Isaac LAN 交付 | [交接紀錄](../results/repo_handoff_2026-09-22.md)、[遷移清單](ISAAC_SIM_HANDOFF.md)、[影像修復](../results/isaac_image_receive_2026-09-23.md)。 |
| 2026-10-15 主機交付 | [已發佈的 commit](https://github.com/Laiting20021202/social-safety-decision/commit/783b5d986487cd66b3e6f333875c631bbe0f9b64)、[主機執行手冊](history/2026-10-15-host-workflows.md)、[Tidybot 退避結果](../results/tidybot_retreat_2026-09-30.md)。 |
| Docker／ROS 2 重整 | [容器與系統圖](ARCHITECTURE.md)、[Sim-to-Real 接入](DOCKER.md)、[程式索引](CODE_MAP.md)、[Docker 驗收](../results/docker_isaac_2026-11-24.md)。 |

[歷史分支與進度](BRANCHES_AND_PROGRESS.md) 記錄當時的累積分支策略。依各筆驗收的模擬器、資料來源與量測條件解讀結果。
