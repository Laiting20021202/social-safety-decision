import {
  Eye,
  EyeOff,
  Lock,
  Pause,
  Play,
  RotateCcw,
  Save,
  Trash2,
  Unlock
} from "lucide-react";
import { PointerEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";

type DatasetInfo = {
  dataset_id: string;
  name: string;
  revision: string;
  source_url: string;
  cached: boolean;
};

type ScenarioInfo = {
  scenario_id: string;
  frame_count: number;
  duration_sec: number;
  image_width: number;
  image_height: number;
  metadata: Record<string, unknown>;
};

type Point2D = { x: number; y: number };

type VideoInfo = {
  scenario_id: string;
  video_reference: string;
  source: string;
  fps: number;
  duration_sec: number;
  frame_count: number;
  generated: boolean;
};

type ZoneDefinition = {
  zone_id: string;
  scenario_id: string;
  name: string;
  source: "manual" | "manual_fallback" | "robopoint" | "robopoint_sam3" | "cached" | "dataset" | "imported";
  coordinate_type: "image" | "bev" | "robot" | "map";
  polygon: Point2D[];
  mask_rle: null;
  prompt: string;
  source_frame_index: number;
  confidence: number;
  locked: boolean;
  opacity: number;
  image_width: number;
  image_height: number;
  metadata: Record<string, unknown>;
};

type RoadSegmentationResult = {
  scenario_id: string;
  frame_index: number;
  timestamp_sec: number;
  source: string;
  polygon: Point2D[];
  confidence: number;
  prompt: string;
  is_valid: boolean;
  metadata: Record<string, unknown>;
};

type TrackObservation = {
  track_id: number;
  class_name: string;
  timestamp_sec: number;
  frame_index: number;
  mask_polygon: Point2D[];
  bounding_box: [number, number, number, number] | null;
  centroid: Point2D;
  centroid_image: Point2D | null;
  ground_contact_point: Point2D | null;
  bottom_center: Point2D | null;
  confidence: number;
  track_age_sec: number;
  lost_count: number;
  metadata: Record<string, unknown>;
};

type MotionEstimate = {
  track_id: number;
  timestamp_sec: number;
  velocity_vector: [number, number] | null;
  speed: number;
  speed_unit: "m/s" | "normalized/s" | "px/s";
  direction_angle_deg: number;
  direction_label_geometry: string;
  direction_label_vqa: string;
  direction_label_fused: string;
  confidence: number;
  is_approximate: boolean;
};

type DynamicRiskZone = {
  track_id: number;
  class_name: string;
  timestamp_sec: number;
  prediction_horizon_sec: number;
  predicted_points: Point2D[];
  risk_polygon: Point2D[];
  speed: number;
  direction: string;
  uncertainty: number;
  intersects_robot_corridor: boolean;
  risk_level: "low" | "warning" | "critical" | "unknown";
  time_to_intersection_sec: number | null;
};

type RobotCorridor = {
  polygon: Point2D[];
  origin: Point2D | null;
  heading_vector: Point2D | null;
  is_approximate: boolean;
  metadata: Record<string, unknown>;
};

type VqaDirection = {
  track_id: number;
  direction_label: string;
  path_relation: string;
  confidence: number;
  updated_at_sec: number | null;
  parse_valid: boolean;
};

type AnalysisPacket = {
  scenario_id: string;
  video_timestamp_sec: number;
  analysis_timestamp_sec: number;
  road: RoadSegmentationResult;
  tracks: TrackObservation[];
  motions: MotionEstimate[];
  vqa_directions: VqaDirection[];
  risk_zones: DynamicRiskZone[];
  robot_corridor: RobotCorridor;
  system_status: {
    tracking_fps: number;
    vqa_update_interval_sec: number;
    analysis_delay_ms: number;
    analysis_age_ms: number;
    vqa_last_update_sec: number;
    tracking_status: string;
    road_status: string;
    vqa_status: string;
    message: string;
  };
  metadata: Record<string, unknown>;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const HORIZONS = [1, 2, 3, 5];
const VQA_INTERVALS = [1, 2, 3, 5];
const BEV_WIDTH = 1000;
const BEV_HEIGHT = 420;

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<T>;
}

function App() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [scenarioDetails, setScenarioDetails] = useState<ScenarioInfo | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisPacket | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [predictionHorizon, setPredictionHorizon] = useState(3);
  const [vqaInterval, setVqaInterval] = useState(2);
  const [roadDraft, setRoadDraft] = useState<Point2D[]>([]);
  const [roadLocked, setRoadLocked] = useState(true);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [toggles, setToggles] = useState({
    road: true,
    agents: true,
    risk: true,
    labels: true
  });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<SVGSVGElement | null>(null);
  const lastAnalysisFetch = useRef(0);
  const analysisInFlight = useRef(false);

  const scenarioListItem = useMemo(
    () => scenarios.find((scenario) => scenario.scenario_id === selectedScenario) ?? null,
    [scenarios, selectedScenario]
  );
  const selectedScenarioInfo =
    scenarioDetails?.scenario_id === selectedScenario ? scenarioDetails : scenarioListItem;
  const viewWidth = selectedScenarioInfo?.image_width || 1280;
  const viewHeight = selectedScenarioInfo?.image_height || 720;
  const motionByTrack = useMemo(
    () => new Map((analysis?.motions ?? []).map((motion) => [motion.track_id, motion])),
    [analysis]
  );
  const vqaByTrack = useMemo(
    () => new Map((analysis?.vqa_directions ?? []).map((vqa) => [vqa.track_id, vqa])),
    [analysis]
  );
  const riskByTrack = useMemo(
    () => new Map((analysis?.risk_zones ?? []).map((risk) => [risk.track_id, risk])),
    [analysis]
  );
  const analysisDelayed = (analysis?.system_status.analysis_delay_ms ?? 0) > 750;

  useEffect(() => {
    getJson<DatasetInfo[]>("/datasets")
      .then((items) => {
        setDatasets(items);
        if (items[0]) setSelectedDataset(items[0].dataset_id);
      })
      .catch((caught) => setError(String(caught)));
  }, []);

  useEffect(() => {
    if (!selectedDataset) return;
    getJson<ScenarioInfo[]>(`/datasets/${selectedDataset}/scenarios`)
      .then((items) => {
        setScenarios(items);
        if (items[0]) setSelectedScenario(items[0].scenario_id);
      })
      .catch((caught) => setError(String(caught)));
  }, [selectedDataset]);

  useEffect(() => {
    if (!selectedScenario) return;
    getJson<ScenarioInfo>(`/scenarios/${selectedScenario}`)
      .then(setScenarioDetails)
      .catch((caught) => setError(String(caught)));
  }, [selectedScenario]);

  useEffect(() => {
    if (!selectedScenario || !selectedScenarioInfo || selectedScenarioInfo.image_width <= 0) return;
    Promise.all([
      getJson<VideoInfo>(`/scenarios/${selectedScenario}/video-info`),
      getAnalysis(selectedScenario, 0, predictionHorizon, vqaInterval)
    ])
      .then(([nextVideo, nextAnalysis]) => {
        setVideoInfo(nextVideo);
        setAnalysis(nextAnalysis);
        setDuration(nextVideo.duration_sec || selectedScenarioInfo.duration_sec);
        setCurrentTime(0);
        setRoadDraft(nextAnalysis.road.is_valid ? nextAnalysis.road.polygon : []);
        const video = videoRef.current;
        if (video) {
          video.pause();
          video.currentTime = 0;
          video.playbackRate = playbackSpeed;
        }
      })
      .catch((caught) => setError(String(caught)));
  }, [selectedScenario, selectedScenarioInfo, predictionHorizon, vqaInterval, playbackSpeed]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = playbackSpeed;
  }, [playbackSpeed, videoInfo]);

  useEffect(() => {
    let animationFrame = 0;
    const tick = (now: number) => {
      const video = videoRef.current;
      if (video) {
        const time = video.currentTime || 0;
        setCurrentTime(time);
        if (
          selectedScenario &&
          !analysisInFlight.current &&
          now - lastAnalysisFetch.current > 120
        ) {
          analysisInFlight.current = true;
          lastAnalysisFetch.current = now;
          getAnalysis(selectedScenario, time, predictionHorizon, vqaInterval)
            .then((packet) => {
              const latestVideoTime = videoRef.current?.currentTime ?? time;
              if (packet.analysis_timestamp_sec >= latestVideoTime - 1.0) {
                setAnalysis(packet);
              }
            })
            .catch((caught) => setError(String(caught)))
            .finally(() => {
              analysisInFlight.current = false;
            });
        }
      }
      animationFrame = window.requestAnimationFrame(tick);
    };
    animationFrame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [selectedScenario, predictionHorizon, vqaInterval]);

  const videoUrl = videoInfo ? `${API_BASE}${videoInfo.video_reference}` : "";
  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
  const roadPolygon = analysis?.road.is_valid ? analysis.road.polygon : roadDraft;
  const roadPoints = roadPolygon.map((point) => `${point.x},${point.y}`).join(" ");
  const roadSource = analysis?.road.is_valid ? analysis.road.source : "unavailable";
  const sam3Status = formatSam3Message(analysis?.metadata.sam3_message);
  const bevMode = String(analysis?.robot_corridor.metadata.mode ?? "Approximate BEV - RGB-only");
  const hasRoadOverlay = roadPolygon.length >= 3;

  const loadScenario = () => {
    if (!selectedScenario) return;
    getJson<VideoInfo>(`/scenarios/${selectedScenario}/video-info`)
      .then(setVideoInfo)
      .catch((caught) => setError(String(caught)));
  };

  const play = () => {
    videoRef.current?.play().catch((caught) => setError(String(caught)));
  };

  const pause = () => {
    videoRef.current?.pause();
  };

  const restart = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    video.play().catch((caught) => setError(String(caught)));
  };

  const saveRoad = () => {
    if (!selectedScenarioInfo || roadDraft.length < 3) return;
    const road: ZoneDefinition = {
      zone_id: `road-${selectedScenarioInfo.scenario_id}`,
      scenario_id: selectedScenarioInfo.scenario_id,
      name: "SAM3 road fallback",
      source: "manual_fallback",
      coordinate_type: "image",
      polygon: roadDraft,
      mask_rle: null,
      prompt: "Mark the walkable path in front of the robot.",
      source_frame_index: 0,
      confidence: 1,
      locked: roadLocked,
      opacity: 0.32,
      image_width: selectedScenarioInfo.image_width,
      image_height: selectedScenarioInfo.image_height,
      metadata: { editor: "web-gui", semantic_role: "sam3_road_fallback" }
    };
    putJson<ZoneDefinition>(`/road/${selectedScenarioInfo.scenario_id}`, road)
      .then(() => getAnalysis(selectedScenarioInfo.scenario_id, currentTime, predictionHorizon, vqaInterval))
      .then(setAnalysis)
      .catch((caught) => setError(String(caught)));
  };

  const clearRoad = () => {
    if (!selectedScenarioInfo) return;
    deleteJson<{ deleted: boolean }>(`/road/${selectedScenarioInfo.scenario_id}`)
      .then(() => {
        setRoadDraft([]);
        return getAnalysis(selectedScenarioInfo.scenario_id, currentTime, predictionHorizon, vqaInterval);
      })
      .then(setAnalysis)
      .catch((caught) => setError(String(caught)));
  };

  const pointerToImagePoint = (event: PointerEvent<SVGSVGElement>): Point2D | null => {
    const svg = overlayRef.current;
    if (!svg) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const inverse = svg.getScreenCTM()?.inverse();
    if (!inverse) return null;
    const transformed = point.matrixTransform(inverse);
    return {
      x: clamp(transformed.x, 0, viewWidth),
      y: clamp(transformed.y, 0, viewHeight)
    };
  };

  const onOverlayPointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (roadLocked || event.target !== overlayRef.current) return;
    const point = pointerToImagePoint(event);
    if (!point) return;
    setRoadDraft((current) => [...current, point]);
  };

  const onVertexPointerDown = (event: PointerEvent<SVGCircleElement>, index: number) => {
    if (roadLocked) return;
    event.stopPropagation();
    setDragIndex(index);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onOverlayPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (dragIndex === null || roadLocked) return;
    const point = pointerToImagePoint(event);
    if (!point) return;
    setRoadDraft((current) => current.map((item, index) => (index === dragIndex ? point : item)));
  };

  return (
    <main className="app-shell">
      <section className="sidebar left-panel" aria-label="Playback controls">
        <div className="brand-row">
          <div>
            <h1>social-safety-amr</h1>
            <span className={analysisDelayed ? "status-pill delayed" : "status-pill connected"}>
              {analysisDelayed ? "Analysis delayed" : "Analysis ready"}
            </span>
          </div>
        </div>

        <label>
          Dataset
          <select value={selectedDataset} onChange={(event) => setSelectedDataset(event.target.value)}>
            {datasets.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.name}
              </option>
            ))}
          </select>
        </label>

        <label>
          Scenario
          <select value={selectedScenario} onChange={(event) => setSelectedScenario(event.target.value)}>
            {scenarios.map((scenario) => (
              <option key={scenario.scenario_id} value={scenario.scenario_id}>
                {scenario.scenario_id}
              </option>
            ))}
          </select>
        </label>

        <button className="primary-button" onClick={loadScenario}>
          Load
        </button>

        <div className="button-grid">
          <IconButton label="Play" onClick={play}>
            <Play size={18} />
          </IconButton>
          <IconButton label="Pause" onClick={pause}>
            <Pause size={18} />
          </IconButton>
          <IconButton label="Restart" onClick={restart}>
            <RotateCcw size={18} />
          </IconButton>
        </div>

        <label>
          Seek
          <input
            type="range"
            min={0}
            max={Math.max(duration, 0)}
            step={0.01}
            value={Math.min(currentTime, duration || currentTime)}
            onChange={(event) => {
              const video = videoRef.current;
              const time = Number(event.target.value);
              if (video) video.currentTime = time;
              setCurrentTime(time);
            }}
          />
        </label>

        <label>
          Playback speed
          <select value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))}>
            {SPEEDS.map((speed) => (
              <option key={speed} value={speed}>
                {speed}x
              </option>
            ))}
          </select>
        </label>

        <label>
          Prediction horizon
          <select
            value={predictionHorizon}
            onChange={(event) => setPredictionHorizon(Number(event.target.value))}
          >
            {HORIZONS.map((horizon) => (
              <option key={horizon} value={horizon}>
                {horizon}s
              </option>
            ))}
          </select>
        </label>

        <label>
          VQA update interval
          <select value={vqaInterval} onChange={(event) => setVqaInterval(Number(event.target.value))}>
            {VQA_INTERVALS.map((interval) => (
              <option key={interval} value={interval}>
                {interval}s
              </option>
            ))}
          </select>
        </label>

        <div className="panel-block compact">
          <h2>Overlays</h2>
          <OverlayToggle
            label="SAM3 Road"
            enabled={toggles.road}
            onClick={() => setToggles((current) => ({ ...current, road: !current.road }))}
          />
          <OverlayToggle
            label="SAM3 Agents"
            enabled={toggles.agents}
            onClick={() => setToggles((current) => ({ ...current, agents: !current.agents }))}
          />
          <OverlayToggle
            label="Risk"
            enabled={toggles.risk}
            onClick={() => setToggles((current) => ({ ...current, risk: !current.risk }))}
          />
          <OverlayToggle
            label="Labels"
            enabled={toggles.labels}
            onClick={() => setToggles((current) => ({ ...current, labels: !current.labels }))}
          />
        </div>

        <div className="readout-grid">
          <span>Video</span>
          <strong>{isPlaying ? "playing" : "paused"}</strong>
          <span>Timestamp</span>
          <strong>{currentTime.toFixed(2)} s</strong>
          <span>Analysis</span>
          <strong>{(analysis?.analysis_timestamp_sec ?? 0).toFixed(2)} s</strong>
          <span>Tracking FPS</span>
          <strong>{(analysis?.system_status.tracking_fps ?? 0).toFixed(1)}</strong>
          <span>SAM3 Road</span>
          <strong>{roadSource}</strong>
        </div>

        <div className="panel-block compact">
          <h2>SAM3 Road Segmentation</h2>
          <div className="button-grid">
            <IconButton label="Save Fallback Path" onClick={saveRoad}>
              <Save size={18} />
            </IconButton>
            <IconButton label={roadLocked ? "Unlock Fallback Editor" : "Lock Fallback Editor"} onClick={() => setRoadLocked((value) => !value)}>
              {roadLocked ? <Unlock size={18} /> : <Lock size={18} />}
            </IconButton>
            <IconButton label="Clear Fallback Path" onClick={clearRoad}>
              <Trash2 size={18} />
            </IconButton>
          </div>
        </div>

        {error && <div className="error-box">{error}</div>}
      </section>

      <section className="viewer-column" aria-label="Analysis viewer">
        <div className="viewer-toolbar">
          <span>{selectedScenario || "No scenario"}</span>
          <span>{videoInfo?.source ?? "video unavailable"}</span>
          <span>SAM3: {sam3Status}</span>
          <span>BEV: {bevMode}</span>
        </div>

        <div className="video-stage">
          {videoUrl ? (
            <>
              <video
                ref={videoRef}
                src={videoUrl}
                playsInline
                preload="metadata"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onLoadedMetadata={(event) => {
                  setDuration(event.currentTarget.duration || videoInfo?.duration_sec || 0);
                  event.currentTarget.playbackRate = playbackSpeed;
                }}
              />
              <svg
                ref={overlayRef}
                className="overlay-layer"
                viewBox={`0 0 ${viewWidth} ${viewHeight}`}
                onPointerDown={onOverlayPointerDown}
                onPointerMove={onOverlayPointerMove}
                onPointerUp={() => setDragIndex(null)}
              >
                {toggles.road && hasRoadOverlay && (
                  <polygon
                    points={roadPoints}
                    className={analysis?.road.is_valid ? "road-mask" : "road-draft"}
                  />
                )}
                {!analysis?.road.is_valid && toggles.labels && (
                  <text x={24} y={34} className="overlay-warning">
                    Road segmentation unavailable
                  </text>
                )}
                {!roadLocked &&
                  roadDraft.map((point, index) => (
                    <circle
                      key={`${point.x}-${point.y}-${index}`}
                      cx={point.x}
                      cy={point.y}
                      r={Math.max(5, viewWidth / 180)}
                      className="road-vertex"
                      onPointerDown={(event) => onVertexPointerDown(event, index)}
                      onDoubleClick={(event) => {
                        event.stopPropagation();
                        setRoadDraft((current) => current.filter((_, itemIndex) => itemIndex !== index));
                      }}
                    />
                  ))}
                {toggles.agents &&
                  (analysis?.tracks ?? []).map((track) => (
                    <AgentOverlay
                      key={track.track_id}
                      track={track}
                      motion={motionByTrack.get(track.track_id)}
                      risk={riskByTrack.get(track.track_id)}
                      showLabel={toggles.labels}
                      viewWidth={viewWidth}
                      viewHeight={viewHeight}
                    />
                  ))}
              </svg>
            </>
          ) : (
            <div className="empty-stage">No video loaded</div>
          )}
        </div>

        <div className="bev-stage">
          <div className="bev-header">
            <h2>BEV Safety Map</h2>
            <span>{analysis?.system_status.message || "Waiting for analysis"}</span>
          </div>
          <svg viewBox={`0 0 ${BEV_WIDTH} ${BEV_HEIGHT}`} className="bev-canvas">
            <rect x="0" y="0" width={BEV_WIDTH} height={BEV_HEIGHT} className="bev-bg" />
            {toggles.road && hasRoadOverlay && (
              <polygon points={toBevPoints(roadPolygon, viewWidth, viewHeight)} className="bev-road" />
            )}
            {analysis?.robot_corridor.polygon && (
              <polygon points={normalizedPoints(analysis.robot_corridor.polygon)} className="bev-corridor" />
            )}
            {analysis?.robot_corridor.origin && (
              <g>
                <circle
                  cx={analysis.robot_corridor.origin.x * BEV_WIDTH}
                  cy={analysis.robot_corridor.origin.y * BEV_HEIGHT}
                  r={14}
                  className="bev-robot"
                />
                <text
                  x={analysis.robot_corridor.origin.x * BEV_WIDTH - 5}
                  y={analysis.robot_corridor.origin.y * BEV_HEIGHT + 5}
                  className="bev-robot-label"
                >
                  R
                </text>
                <line x1={BEV_WIDTH / 2} y1={BEV_HEIGHT - 22} x2={BEV_WIDTH / 2} y2={80} className="bev-forward" />
              </g>
            )}
            {toggles.risk &&
              (analysis?.risk_zones ?? []).map((risk) => (
                <g key={`risk-${risk.track_id}`}>
                  <polygon
                    points={normalizedPoints(risk.risk_polygon)}
                    className={`bev-risk ${risk.risk_level}`}
                  />
                  <polyline points={normalizedPoints(risk.predicted_points)} className="bev-trajectory" />
                </g>
              ))}
            {toggles.agents &&
              (analysis?.tracks ?? []).map((track) => {
                const point = pointToBev(track.ground_contact_point ?? track.centroid, viewWidth, viewHeight);
                const motion = motionByTrack.get(track.track_id);
                const risk = riskByTrack.get(track.track_id);
                const arrow = motion?.velocity_vector ?? [0, 0];
                return (
                  <g key={`bev-agent-${track.track_id}`}>
                    <circle cx={point.x * BEV_WIDTH} cy={point.y * BEV_HEIGHT} r={9} className="bev-agent" />
                    <line
                      x1={point.x * BEV_WIDTH}
                      y1={point.y * BEV_HEIGHT}
                      x2={(point.x + arrow[0] * 1.2) * BEV_WIDTH}
                      y2={(point.y + arrow[1] * 1.2) * BEV_HEIGHT}
                      className="bev-agent-arrow"
                    />
                    <text x={point.x * BEV_WIDTH + 12} y={point.y * BEV_HEIGHT - 8} className="bev-label">
                      {risk?.intersects_robot_corridor ? `ID ${track.track_id} Collision Risk` : `ID ${track.track_id}`}
                    </text>
                  </g>
                );
              })}
            {analysis && analysis.tracks.length === 0 && (
              <text x={32} y={54} className="bev-empty-note">
                SAM3 tracking unavailable - BEV corridor only
              </text>
            )}
          </svg>
          <div className="timeline">
            <div className="timeline-track">
              <div className="timeline-progress" style={{ width: `${clamp(progress, 0, 100)}%` }} />
            </div>
          </div>
        </div>
      </section>

      <section className="sidebar right-panel" aria-label="Track list">
        <div className="panel-block">
          <h2>Track List</h2>
          <div className="track-table">
            <div className="track-row header">
              <span>ID</span>
              <span>Type</span>
              <span>Direction</span>
              <span>Speed</span>
              <span>Risk</span>
            </div>
            {(analysis?.tracks ?? []).map((track) => {
              const motion = motionByTrack.get(track.track_id);
              const vqa = vqaByTrack.get(track.track_id);
              const risk = riskByTrack.get(track.track_id);
              return (
                <details key={track.track_id} className="track-details" open={risk?.risk_level === "critical"}>
                  <summary className="track-row">
                    <span>{track.track_id}</span>
                    <span>{track.class_name}</span>
                    <span>{formatDirection(motion?.direction_label_fused ?? "uncertain")}</span>
                    <span>{formatSpeed(motion)}</span>
                    <span className={`risk-chip ${risk?.risk_level ?? "unknown"}`}>{risk?.risk_level ?? "unknown"}</span>
                  </summary>
                  <StatusRow label="Path relation" value={vqa?.path_relation ?? "uncertain"} />
                  <StatusRow label="VQA update" value={vqa?.updated_at_sec != null ? `${vqa.updated_at_sec.toFixed(1)} s` : "unavailable"} />
                  <StatusRow label="VQA confidence" value={(vqa?.confidence ?? 0).toFixed(2)} />
                  <StatusRow label="Motion cue" value={formatDirection(motion?.direction_label_geometry ?? "uncertain")} />
                  <StatusRow label="Track confidence" value={track.confidence.toFixed(2)} />
                  <StatusRow label="Time to path" value={risk?.time_to_intersection_sec != null ? `${risk.time_to_intersection_sec.toFixed(1)} s` : "none"} />
                </details>
              );
            })}
            {analysis && analysis.tracks.length === 0 && (
              <div className="empty-table">Tracking unavailable</div>
            )}
          </div>
        </div>

        <div className="panel-block">
          <h2>System Status</h2>
          <StatusRow label="SAM3 road" value={analysis?.system_status.road_status ?? "unavailable"} />
          <StatusRow label="SAM3 tracking" value={analysis?.system_status.tracking_status ?? "unavailable"} />
          <StatusRow label="VQA" value={analysis?.system_status.vqa_status ?? "unavailable"} />
          <StatusRow label="Delay" value={`${analysis?.system_status.analysis_delay_ms ?? 0} ms`} />
          <StatusRow label="BEV safety map" value={bevMode} />
          <StatusRow label="SAM3 runtime" value={sam3Status} />
        </div>

        <details className="debug-drawer">
          <summary>Debug JSON</summary>
          <pre>{analysis ? JSON.stringify(analysis, null, 2) : "{}"}</pre>
        </details>
      </section>
    </main>
  );
}

async function getAnalysis(
  scenarioId: string,
  timestampSec: number,
  predictionHorizonSec: number,
  vqaUpdateIntervalSec: number
) {
  const params = new URLSearchParams({
    timestamp_sec: timestampSec.toFixed(3),
    prediction_horizon_sec: String(predictionHorizonSec),
    vqa_update_interval_sec: String(vqaUpdateIntervalSec)
  });
  return getJson<AnalysisPacket>(`/scenarios/${scenarioId}/analysis?${params.toString()}`);
}

function AgentOverlay({
  track,
  motion,
  risk,
  showLabel,
  viewWidth,
  viewHeight
}: {
  track: TrackObservation;
  motion?: MotionEstimate;
  risk?: DynamicRiskZone;
  showLabel: boolean;
  viewWidth: number;
  viewHeight: number;
}) {
  const bbox = track.bounding_box;
  const ground = track.ground_contact_point ?? track.bottom_center ?? track.centroid;
  const velocity = motion?.velocity_vector ?? [0, 0];
  const arrowScale = Math.min(1.2, Math.max(0.35, (motion?.speed ?? 0.04) * 2.8));
  const x2 = ground.x + velocity[0] * viewWidth * arrowScale;
  const y2 = ground.y + velocity[1] * viewHeight * arrowScale;
  return (
    <g className="agent-overlay">
      {track.mask_polygon.length >= 3 && (
        <polygon
          points={track.mask_polygon.map((point) => `${point.x},${point.y}`).join(" ")}
          className={`agent-segmentation ${track.class_name}`}
        />
      )}
      {bbox && (
        <rect
          x={bbox[0]}
          y={bbox[1]}
          width={Math.max(1, bbox[2] - bbox[0])}
          height={Math.max(1, bbox[3] - bbox[1])}
          className={`agent-mask ${track.class_name}`}
        />
      )}
      <circle cx={ground.x} cy={ground.y} r={7} className="ground-point" />
      <line x1={ground.x} y1={ground.y} x2={x2} y2={y2} className="direction-arrow" />
      {showLabel && (
        <g>
          <rect x={ground.x + 12} y={ground.y - 74} width={220} height={64} rx={6} className="agent-label-bg" />
          <text x={ground.x + 22} y={ground.y - 52} className="agent-label">
            ID {track.track_id} | {titleCase(track.class_name)}
          </text>
          <text x={ground.x + 22} y={ground.y - 32} className="agent-label-sub">
            Direction: {formatDirection(motion?.direction_label_fused ?? "uncertain")}
          </text>
          <text x={ground.x + 22} y={ground.y - 14} className="agent-label-sub">
            Speed: {formatSpeed(motion)} | Risk: {risk?.risk_level ?? "unknown"}
          </text>
        </g>
      )}
    </g>
  );
}

function IconButton({
  label,
  onClick,
  children
}: {
  label: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button className="icon-button" onClick={onClick} title={label} aria-label={label}>
      {children}
    </button>
  );
}

function OverlayToggle({
  label,
  enabled,
  onClick
}: {
  label: string;
  enabled: boolean;
  onClick: () => void;
}) {
  return (
    <button className={enabled ? "overlay-toggle active" : "overlay-toggle"} onClick={onClick}>
      {enabled ? <Eye size={15} /> : <EyeOff size={15} />}
      <span>{label}</span>
    </button>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function pointToBev(point: Point2D, imageWidth: number, imageHeight: number): Point2D {
  const nx = imageWidth > 0 ? clamp(point.x / imageWidth, 0, 1) : 0.5;
  const ny = imageHeight > 0 ? clamp(point.y / imageHeight, 0, 1) : 0.5;
  return { x: nx, y: clamp((ny - 0.45) / 0.55, 0, 1) };
}

function toBevPoints(points: Point2D[], imageWidth: number, imageHeight: number) {
  return points
    .map((point) => {
      const bev = pointToBev(point, imageWidth, imageHeight);
      return `${bev.x * BEV_WIDTH},${bev.y * BEV_HEIGHT}`;
    })
    .join(" ");
}

function normalizedPoints(points: Point2D[]) {
  return points.map((point) => `${point.x * BEV_WIDTH},${point.y * BEV_HEIGHT}`).join(" ");
}

function formatSpeed(motion?: MotionEstimate) {
  if (!motion) return "unavailable";
  const suffix = motion.is_approximate ? " approx" : "";
  return `${motion.speed.toFixed(2)} ${motion.speed_unit}${suffix}`;
}

function formatDirection(direction: string) {
  return direction.replace(/_/g, " ");
}

function formatSam3Message(value: unknown) {
  const raw = String(value ?? "");
  if (!raw) return "not configured";
  let message = raw;
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (parsed.detail) message = String(parsed.detail);
  } catch {
    message = raw;
  }
  if (message.includes("gated repo") || message.includes("Access to model facebook/sam3 is restricted")) {
    return "gated model - HF_TOKEN required";
  }
  if (message.includes("SAM3 segmentation available")) return "available";
  if (message.includes("SAM3_SERVICE_URL is not configured")) return "service not configured";
  if (message.length > 92) return `${message.slice(0, 89)}...`;
  return message;
}

function titleCase(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export default App;
