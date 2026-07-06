# RGB Social Navigation BEV System Architecture

## System Architecture

```mermaid
flowchart TB
  %% =========================
  %% User interfaces
  %% =========================
  subgraph UI["User Interfaces"]
    GUI["Tkinter GUI\n- Start / Pause / Stop\n- Realtime playback\n- Live 2x2 preview"]
    CLI["CLI\npython -m social_bev.run"]
    Streamlit["Optional Streamlit UI\napp.py"]
    ROS2["Optional ROS2 Adapter\nros2/social_bev_node.py"]
  end

  %% =========================
  %% Inputs
  %% =========================
  subgraph INPUT["RGB Input Sources"]
    Webcam["Realtime camera\nInput = 0"]
    Video["Local video\nMP4 / AVI / MOV / MKV"]
    Images["Image directory\nSCAND sample / frames"]
    SingleImage["Single image"]
  end

  GUI --> FrameSource
  CLI --> FrameSource
  Streamlit --> FrameSource
  ROS2 --> Pipeline

  Webcam --> FrameSource
  Video --> FrameSource
  Images --> FrameSource
  SingleImage --> FrameSource

  %% =========================
  %% Frame source
  %% =========================
  subgraph SOURCE["Frame Source Layer"]
    FrameSource["FrameSource\n- Decode frames\n- Timestamp\n- Stride\n- OpenCV VideoCapture"]
    FFMPEG["ffmpeg fallback\nwhen OpenCV cannot decode video"]
  end

  FrameSource -->|OpenCV fails| FFMPEG
  FFMPEG --> Frame
  FrameSource --> Frame["BGR frame + timestamp"]

  %% =========================
  %% Runtime control
  %% =========================
  subgraph CONTROL["Runtime Control"]
    Realtime["Realtime pacing\navoid precomputing full video"]
    Pause["Pause/Resume gate\nstop before next frame enters inference"]
    Config["configs/default.yaml\nresolution, intervals, thresholds"]
    Calibration["configs/calibration.yaml\nmetric homography if available"]
  end

  GUI --> Realtime
  GUI --> Pause
  Realtime --> Pipeline
  Pause --> Pipeline
  Config --> Pipeline
  Calibration --> Homography
  Frame --> Pipeline

  %% =========================
  %% Core pipeline
  %% =========================
  subgraph Pipeline["SocialNavigationPipeline CPU Core"]
    Resize["Resize + preprocess\ninput_width x input_height"]

    subgraph Perception["RGB Perception"]
      Seg["Walkable segmentation\nSegFormer Torch/OpenVINO\nfallback: RGB heuristic"]
      Det["Object detection\nYOLO11 CPU/OpenVINO\nfallback: OpenCV HOG"]
      Track["Person tracking\nKalman filter + Hungarian matching"]
      Unknown["Unknown obstacle extractor\nnon-walkable RGB regions\ninside ground corridor"]
    end

    Ground["Ground contact estimation\nperson feet / obstacle bottom points"]
    Homography["Homography / IPM projection\nimage points -> BEV pixels"]
    BEV["BEVMapBuilder\nfree space, people, obstacles, robot"]
    Social["Social zone rendering\nstatic or motion-elongated zones"]
    Viz["2x2 visualization composer\nfront view, mask, contacts, BEV"]
    JSON["Frame result object\ntracks, detections, timing, FPS"]
  end

  Pipeline --> Resize
  Resize --> Seg
  Resize --> Det
  Seg --> Unknown
  Seg --> Ground
  Det --> Track
  Det --> Unknown
  Det --> Ground
  Track --> Ground
  Ground --> Homography
  Unknown --> Homography
  Seg --> Homography
  Homography --> BEV
  BEV --> Social
  Social --> Viz
  Track --> JSON
  Det --> JSON
  Unknown --> JSON
  Social --> JSON

  %% =========================
  %% Outputs
  %% =========================
  subgraph OUTPUT["Outputs"]
    Preview["Live GUI preview\n2x2 result"]
    MP4["Output video\noutputs/*.mp4"]
    JSONL["Per-frame JSONL\npeople, known/unknown obstacles,\nprocessing_ms, FPS"]
    Occupancy["Occupancy grid\n.npy exact values\n.png grayscale preview"]
    ROSOut["ROS2 topics\n/social_bev/annotated\n/social_bev/walkable_mask\n/social_bev/occupancy_grid\n/social_bev/people_markers"]
  end

  Viz --> Preview
  Viz --> MP4
  JSON --> JSONL
  BEV --> Occupancy
  ROS2 --> ROSOut

  %% =========================
  %% Legend
  %% =========================
  subgraph LEGEND["Occupancy Grid Values"]
    L1["-1 unknown"]
    L2["0 free"]
    L3["50 social caution"]
    L4["80 unknown obstacle"]
    L5["100 occupied"]
  end
```

## Notes

- The system is online frame-by-frame processing. It does not need to precompute the full video.
- GUI pause stops the next frame before it enters the inference pipeline; it cannot interrupt the currently running inference call.
- Realtime camera input is supported by setting the input to `0`, but CPU inference speed is not guaranteed to match the camera FPS.
- Without `configs/calibration.yaml`, BEV is non-metric and displayed as `NON-METRIC BEV`.
- This project is for perception and risk visualization, not a safety-certified robot controller.
