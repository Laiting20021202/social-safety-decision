from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from realtime_safety.edgetam_tracker.models import (
    AABB,
    Cluster3D,
    MaskQuality,
    OBB,
    PointCloudQuality,
    TrackEstimate,
    TrackingState,
)


MaskIoUs = Mapping[tuple[int, int], float] | np.ndarray | None


def _validated_robot_origin(
    value: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    try:
        origin = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("robot_origin must contain exactly three finite values") from exc
    if origin.size != 3:
        raise ValueError("robot_origin must contain exactly three finite values")
    origin = origin.reshape(3)
    if not np.isfinite(origin).all():
        raise ValueError("robot_origin must contain exactly three finite values")
    return origin.astype(np.float32)


@dataclass(slots=True)
class PointCloudTrackerConfig:
    confirmation_hits: int = 3
    emergency_confirmation_distance: float = 0.35
    emergency_confidence_floor: float = 0.25
    maximum_association_distance: float = 0.75
    maximum_mahalanobis_distance: float = 6.0
    maximum_size_log_difference: float = 1.6
    maximum_count_log_difference: float = 2.5
    minimum_mask_iou: float = 0.0
    maximum_association_cost: float = 0.82
    centroid_cost_weight: float = 0.55
    aabb_size_cost_weight: float = 0.18
    point_count_cost_weight: float = 0.12
    mask_iou_cost_weight: float = 0.15
    initial_position_variance: float = 0.05
    initial_velocity_variance: float = 1.0
    acceleration_process_variance: float = 1.0
    measurement_variance: float = 0.025
    measured_velocity_blend: float = 0.30
    geometry_smoothing_alpha: float = 0.65
    maximum_occluded_frames: int = 3
    occluded_retention_seconds: float = 0.5
    maximum_missed_frames: int = 30
    lost_retention_seconds: float = 2.0
    maximum_tentative_misses: int = 1
    tentative_retention_seconds: float = 0.35
    maximum_prediction_age_seconds: float = 1.0
    prediction_horizon_seconds: float = 3.0
    prediction_step_seconds: float = 0.2
    prediction_horizons_seconds: tuple[float, ...] | None = None
    missed_confidence_decay: float = 0.82
    uncertainty_covariance_gain: float = 0.20
    lost_uncertainty_gain_per_second: float = 0.10
    robot_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be at least one")
        if self.maximum_association_distance <= 0.0:
            raise ValueError("maximum_association_distance must be positive")
        if self.measurement_variance <= 0.0:
            raise ValueError("measurement_variance must be positive")
        if self.prediction_step_seconds <= 0.0:
            raise ValueError("prediction_step_seconds must be positive")
        if self.prediction_horizon_seconds < 0.0:
            raise ValueError("prediction_horizon_seconds cannot be negative")
        if self.prediction_horizons_seconds is not None:
            horizons = tuple(
                sorted(
                    {
                        float(value)
                        for value in self.prediction_horizons_seconds
                        if float(value) > 0.0
                    }
                )
            )
            if not all(np.isfinite(value) for value in horizons):
                raise ValueError("prediction horizons must be finite")
            self.prediction_horizons_seconds = horizons
        if self.maximum_prediction_age_seconds < 0.0:
            raise ValueError("maximum_prediction_age_seconds cannot be negative")
        origin = _validated_robot_origin(self.robot_origin)
        self.robot_origin = tuple(float(value) for value in origin)


@dataclass(slots=True)
class _Track:
    track_id: int
    state: TrackingState
    x: np.ndarray
    covariance: np.ndarray
    aabb: AABB
    obb: OBB
    nearest_point: np.ndarray
    nearest_distance: float
    point_count: int
    hit_count: int
    consecutive_hits: int
    missed_count: int
    age_frames: int
    first_timestamp: float
    last_measurement_timestamp: float
    filter_timestamp: float
    last_measurement_position: np.ndarray
    confidence: float
    pointcloud_quality: PointCloudQuality
    pointcloud_quality_score: float
    mask_quality: MaskQuality = MaskQuality.UNAVAILABLE
    mask_quality_score: float = 0.0
    source_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    source_colors: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    last_association_cost: float = 0.0
    ever_confirmed: bool = False
    emergency_promoted: bool = False


class PointCloudTracker:
    """One-to-one 3D association backed by six-state CV Kalman filters."""

    def __init__(self, config: PointCloudTrackerConfig | None = None) -> None:
        self.config = config or PointCloudTrackerConfig()
        self._robot_origin = _validated_robot_origin(self.config.robot_origin)
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp = None

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    @property
    def robot_origin(self) -> np.ndarray:
        """Return a copy of the origin currently used for distance estimates."""

        return self._robot_origin.copy()

    def set_robot_origin(
        self,
        robot_origin: tuple[float, float, float] | np.ndarray,
    ) -> None:
        """Atomically replace the finite 3D origin used by all tracks."""

        origin = _validated_robot_origin(robot_origin)
        self._robot_origin = origin
        self.config.robot_origin = tuple(float(value) for value in origin)
        for track in self._tracks.values():
            track.nearest_point, track.nearest_distance = self._nearest_geometry(
                track.source_points,
                track.nearest_point,
            )

    def update(
        self,
        clusters: list[Cluster3D],
        timestamp: float,
        mask_ious: MaskIoUs = None,
    ) -> list[TrackEstimate]:
        """Predict, associate once, update lifecycle, and emit current tracks."""

        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("tracker timestamps must be nondecreasing")
        self._last_timestamp = timestamp

        # DELETED is observable for one update and removed before the next one.
        for track_id in [
            key
            for key, track in self._tracks.items()
            if track.state is TrackingState.DELETED
        ]:
            del self._tracks[track_id]

        active = [
            track
            for track in sorted(
                self._tracks.values(), key=lambda item: item.track_id
            )
            if track.state is not TrackingState.DELETED
        ]
        for track in active:
            self._predict_track(track, timestamp)
            track.age_frames += 1

        matches, unmatched_tracks, unmatched_clusters = self._associate(
            active,
            clusters,
            mask_ious,
        )
        for track_index, cluster_index, cost in matches:
            self._update_track(
                active[track_index],
                clusters[cluster_index],
                timestamp,
                cost,
            )
        for track_index in unmatched_tracks:
            self._mark_missed(active[track_index], timestamp)
        for cluster_index in unmatched_clusters:
            track = self._new_track(clusters[cluster_index], timestamp)
            self._tracks[track.track_id] = track

        return [
            self._to_estimate(track)
            for track in sorted(
                self._tracks.values(), key=lambda item: item.track_id
            )
        ]

    def predict_to(
        self,
        timestamp: float,
        *,
        include_deleted: bool = False,
    ) -> list[TrackEstimate]:
        """Return non-mutating predictions bounded by the configured age."""

        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        estimates: list[TrackEstimate] = []
        for track in sorted(
            self._tracks.values(), key=lambda item: item.track_id
        ):
            if track.state is TrackingState.DELETED and not include_deleted:
                continue
            x, covariance, offset = self._predicted_copy(track, timestamp)
            estimates.append(
                self._to_estimate(
                    track,
                    x=x,
                    covariance=covariance,
                    geometry_offset=offset,
                    filter_timestamp=max(timestamp, track.filter_timestamp),
                )
            )
        return estimates

    def _associate(
        self,
        tracks: list[_Track],
        clusters: list[Cluster3D],
        mask_ious: MaskIoUs,
    ) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
        if not tracks or not clusters:
            return [], set(range(len(tracks))), set(range(len(clusters)))

        costs = np.full((len(tracks), len(clusters)), np.inf, dtype=np.float64)
        for track_index, track in enumerate(tracks):
            for cluster_index, cluster in enumerate(clusters):
                costs[track_index, cluster_index] = self._association_cost(
                    track,
                    cluster,
                    self._lookup_mask_iou(
                        mask_ious,
                        track,
                        cluster,
                        track_index,
                        cluster_index,
                        len(tracks),
                        len(clusters),
                    ),
                )

        finite_costs = np.isfinite(costs)
        if not finite_costs.any():
            return [], set(range(len(tracks))), set(range(len(clusters)))
        sentinel = max(self.config.maximum_association_cost + 1.0, 1e6)
        rows, columns = linear_sum_assignment(
            np.where(finite_costs, costs, sentinel)
        )
        matches: list[tuple[int, int, float]] = []
        matched_tracks: set[int] = set()
        matched_clusters: set[int] = set()
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            cost = float(costs[row, column])
            if (
                np.isfinite(cost)
                and cost <= self.config.maximum_association_cost
            ):
                matches.append((row, column, cost))
                matched_tracks.add(row)
                matched_clusters.add(column)
        return (
            matches,
            set(range(len(tracks))) - matched_tracks,
            set(range(len(clusters))) - matched_clusters,
        )

    def _association_cost(
        self,
        track: _Track,
        cluster: Cluster3D,
        mask_iou: float | None,
    ) -> float:
        innovation = (
            np.asarray(cluster.centroid, dtype=np.float64) - track.x[:3]
        )
        centroid_distance = float(np.linalg.norm(innovation))
        if centroid_distance > self.config.maximum_association_distance:
            return float("inf")

        measurement_covariance = (
            track.covariance[:3, :3]
            + np.eye(3, dtype=np.float64)
            * self.config.measurement_variance
        )
        try:
            mahalanobis = float(
                np.sqrt(
                    max(
                        innovation
                        @ np.linalg.solve(
                            measurement_covariance,
                            innovation,
                        ),
                        0.0,
                    )
                )
            )
        except np.linalg.LinAlgError:
            mahalanobis = float("inf")
        if mahalanobis > self.config.maximum_mahalanobis_distance:
            return float("inf")

        current_size = np.maximum(
            np.asarray(track.aabb.size, dtype=np.float64),
            1e-4,
        )
        candidate_size = np.maximum(
            np.asarray(cluster.aabb.size, dtype=np.float64),
            1e-4,
        )
        size_log_difference = float(
            np.mean(np.abs(np.log(candidate_size / current_size)))
        )
        if size_log_difference > self.config.maximum_size_log_difference:
            return float("inf")

        count_log_difference = float(
            abs(
                np.log(
                    (max(int(cluster.point_count), 0) + 1.0)
                    / (max(track.point_count, 0) + 1.0)
                )
            )
        )
        if count_log_difference > self.config.maximum_count_log_difference:
            return float("inf")
        if (
            mask_iou is not None
            and mask_iou < self.config.minimum_mask_iou
        ):
            return float("inf")

        terms = [
            (
                centroid_distance
                / self.config.maximum_association_distance,
                self.config.centroid_cost_weight,
            ),
            (
                size_log_difference
                / max(self.config.maximum_size_log_difference, 1e-6),
                self.config.aabb_size_cost_weight,
            ),
            (
                count_log_difference
                / max(self.config.maximum_count_log_difference, 1e-6),
                self.config.point_count_cost_weight,
            ),
        ]
        if mask_iou is not None:
            terms.append(
                (
                    1.0 - float(np.clip(mask_iou, 0.0, 1.0)),
                    self.config.mask_iou_cost_weight,
                )
            )
        weight = sum(max(item_weight, 0.0) for _, item_weight in terms)
        if weight <= 1e-12:
            return centroid_distance
        return float(
            sum(value * max(item_weight, 0.0) for value, item_weight in terms)
            / weight
        )

    @staticmethod
    def _lookup_mask_iou(
        mask_ious: MaskIoUs,
        track: _Track,
        cluster: Cluster3D,
        track_index: int,
        cluster_index: int,
        track_count: int,
        cluster_count: int,
    ) -> float | None:
        if mask_ious is None:
            return None
        if isinstance(mask_ious, Mapping):
            value = mask_ious.get((track.track_id, cluster.cluster_id))
            return None if value is None else float(value)
        matrix = np.asarray(mask_ious, dtype=np.float64)
        if matrix.shape != (track_count, cluster_count):
            raise ValueError(
                "mask_ious array must have shape (track_count, cluster_count)"
            )
        value = float(matrix[track_index, cluster_index])
        return value if np.isfinite(value) else None

    def _new_track(self, cluster: Cluster3D, timestamp: float) -> _Track:
        track_id = self._next_track_id
        self._next_track_id += 1
        position = np.asarray(cluster.centroid, dtype=np.float64).reshape(3)
        source_points = cluster.points.astype(np.float32).copy()
        nearest_point, nearest_distance = self._nearest_geometry(
            source_points,
            cluster.nearest_point,
        )
        x = np.r_[position, np.zeros(3, dtype=np.float64)]
        covariance = np.diag(
            [self.config.initial_position_variance] * 3
            + [self.config.initial_velocity_variance] * 3
        ).astype(np.float64)
        emergency = (
            self.config.emergency_confirmation_distance > 0.0
            and nearest_distance
            <= self.config.emergency_confirmation_distance
        )
        confirmed = self.config.confirmation_hits <= 1 or emergency
        confidence = float(np.clip(cluster.quality_score, 0.0, 1.0))
        if emergency:
            confidence = max(
                confidence,
                self.config.emergency_confidence_floor,
            )
        return _Track(
            track_id=track_id,
            state=(
                TrackingState.CONFIRMED
                if confirmed
                else TrackingState.TENTATIVE
            ),
            x=x,
            covariance=covariance,
            aabb=AABB(cluster.aabb.minimum.copy(), cluster.aabb.maximum.copy()),
            obb=OBB(
                cluster.obb.center.copy(),
                cluster.obb.size.copy(),
                cluster.obb.rotation.copy(),
            ),
            nearest_point=nearest_point,
            nearest_distance=nearest_distance,
            point_count=int(cluster.point_count),
            hit_count=1,
            consecutive_hits=1,
            missed_count=0,
            age_frames=1,
            first_timestamp=timestamp,
            last_measurement_timestamp=timestamp,
            filter_timestamp=timestamp,
            last_measurement_position=position.copy(),
            confidence=confidence,
            pointcloud_quality=cluster.quality,
            pointcloud_quality_score=float(cluster.quality_score),
            source_points=source_points,
            source_colors=(
                None
                if cluster.colors is None
                else cluster.colors.astype(np.uint8).copy()
            ),
            source_indices=(
                None
                if cluster.source_indices is None
                else cluster.source_indices.astype(np.int64).copy()
            ),
            ever_confirmed=confirmed,
            emergency_promoted=emergency,
        )

    def _update_track(
        self,
        track: _Track,
        cluster: Cluster3D,
        timestamp: float,
        association_cost: float,
    ) -> None:
        measurement = np.asarray(cluster.centroid, dtype=np.float64).reshape(3)
        measurement_dt = max(
            timestamp - track.last_measurement_timestamp,
            1e-6,
        )
        measured_velocity = (
            measurement - track.last_measurement_position
        ) / measurement_dt

        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3)
        quality = max(float(cluster.quality_score), 0.05)
        measurement_variance = self.config.measurement_variance / quality
        noise = np.eye(3, dtype=np.float64) * measurement_variance
        residual = measurement - observation @ track.x
        innovation = (
            observation @ track.covariance @ observation.T + noise
        )
        try:
            gain = (
                track.covariance
                @ observation.T
                @ np.linalg.inv(innovation)
            )
        except np.linalg.LinAlgError:
            gain = (
                track.covariance
                @ observation.T
                @ np.linalg.pinv(innovation)
            )
        track.x = track.x + gain @ residual
        identity = np.eye(6, dtype=np.float64)
        correction = identity - gain @ observation
        track.covariance = (
            correction @ track.covariance @ correction.T
            + gain @ noise @ gain.T
        )
        velocity_blend = float(
            np.clip(self.config.measured_velocity_blend, 0.0, 1.0)
        )
        track.x[3:] = (
            (1.0 - velocity_blend) * track.x[3:]
            + velocity_blend * measured_velocity
        )

        # The Kalman state filters centroid/velocity, not measured safety
        # geometry. Translating a real surface by the filter residual can move
        # an approaching obstacle farther from the robot and overestimate
        # clearance. Keep every matched high-resolution point, AABB, OBB, and
        # nearest surface sample in the actual measurement coordinates.
        track.aabb = AABB(
            cluster.aabb.minimum.copy(),
            cluster.aabb.maximum.copy(),
        )
        track.obb = OBB(
            cluster.obb.center.copy(),
            cluster.obb.size.copy(),
            cluster.obb.rotation.copy(),
        )
        track.source_points = np.asarray(
            cluster.points,
            dtype=np.float32,
        ).copy()
        track.nearest_point, track.nearest_distance = self._nearest_geometry(
            track.source_points,
            cluster.nearest_point,
        )
        track.source_colors = (
            None
            if cluster.colors is None
            else cluster.colors.astype(np.uint8).copy()
        )
        track.source_indices = (
            None
            if cluster.source_indices is None
            else cluster.source_indices.astype(np.int64).copy()
        )
        track.point_count = int(cluster.point_count)
        track.pointcloud_quality = cluster.quality
        track.pointcloud_quality_score = float(cluster.quality_score)
        track.hit_count += 1
        track.consecutive_hits += 1
        track.missed_count = 0
        track.last_measurement_timestamp = timestamp
        track.last_measurement_position = measurement.copy()
        track.last_association_cost = float(association_cost)

        emergency = (
            self.config.emergency_confirmation_distance > 0.0
            and track.nearest_distance
            <= self.config.emergency_confirmation_distance
        )
        if emergency:
            track.emergency_promoted = True
        if (
            track.ever_confirmed
            or track.consecutive_hits >= self.config.confirmation_hits
            or emergency
        ):
            track.state = TrackingState.CONFIRMED
            track.ever_confirmed = True
        else:
            track.state = TrackingState.TENTATIVE

        temporal_score = min(
            track.consecutive_hits / max(self.config.confirmation_hits, 1),
            1.0,
        )
        track.confidence = float(
            np.clip(
                0.65 * track.pointcloud_quality_score
                + 0.35 * temporal_score,
                0.0,
                1.0,
            )
        )
        if emergency:
            track.confidence = max(
                track.confidence,
                self.config.emergency_confidence_floor,
            )

    def _mark_missed(self, track: _Track, timestamp: float) -> None:
        track.missed_count += 1
        track.consecutive_hits = 0
        # Cached/predicted geometry remains available for fail-safe output,
        # but it is not a current point-cloud measurement.
        track.pointcloud_quality = PointCloudQuality.INVALID
        track.pointcloud_quality_score = 0.0
        elapsed = max(timestamp - track.last_measurement_timestamp, 0.0)
        track.confidence *= float(
            np.clip(self.config.missed_confidence_decay, 0.0, 1.0)
        )

        if not track.ever_confirmed:
            if (
                track.missed_count <= self.config.maximum_tentative_misses
                and elapsed <= self.config.tentative_retention_seconds
            ):
                track.state = TrackingState.LOST
            else:
                track.state = TrackingState.DELETED
            return

        if (
            track.missed_count <= self.config.maximum_occluded_frames
            and elapsed <= self.config.occluded_retention_seconds
        ):
            track.state = TrackingState.OCCLUDED
        elif (
            track.missed_count <= self.config.maximum_missed_frames
            and elapsed <= self.config.lost_retention_seconds
        ):
            track.state = TrackingState.LOST
        else:
            track.state = TrackingState.DELETED

    def _predict_track(self, track: _Track, timestamp: float) -> None:
        if timestamp <= track.filter_timestamp:
            return
        allowed = max(
            self.config.maximum_prediction_age_seconds
            - max(
                track.filter_timestamp - track.last_measurement_timestamp,
                0.0,
            ),
            0.0,
        )
        dt = min(timestamp - track.filter_timestamp, allowed)
        if dt > 0.0:
            previous_position = track.x[:3].copy()
            transition, process_noise = self._transition(dt)
            track.x = transition @ track.x
            track.covariance = (
                transition @ track.covariance @ transition.T + process_noise
            )
            offset = track.x[:3] - previous_position
            track.aabb = track.aabb.translated(offset)
            track.obb = track.obb.translated(offset)
            if len(track.source_points):
                track.source_points = (
                    track.source_points + offset.astype(np.float32)
                )
            track.nearest_point, track.nearest_distance = self._nearest_geometry(
                track.source_points,
                track.nearest_point + offset.astype(np.float32),
            )
        track.filter_timestamp = timestamp

    def _predicted_copy(
        self,
        track: _Track,
        timestamp: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = track.x.copy()
        covariance = track.covariance.copy()
        if timestamp <= track.filter_timestamp:
            return x, covariance, np.zeros(3, dtype=np.float64)
        allowed = max(
            self.config.maximum_prediction_age_seconds
            - max(
                track.filter_timestamp - track.last_measurement_timestamp,
                0.0,
            ),
            0.0,
        )
        dt = min(timestamp - track.filter_timestamp, allowed)
        if dt <= 0.0:
            return x, covariance, np.zeros(3, dtype=np.float64)
        transition, process_noise = self._transition(dt)
        predicted = transition @ x
        covariance = (
            transition @ covariance @ transition.T + process_noise
        )
        return predicted, covariance, predicted[:3] - x[:3]

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        q = max(self.config.acceleration_process_variance, 0.0)
        process = np.zeros((6, 6), dtype=np.float64)
        position_variance = 0.25 * dt**4 * q
        cross_variance = 0.5 * dt**3 * q
        velocity_variance = dt**2 * q
        process[:3, :3] = np.eye(3) * position_variance
        process[:3, 3:] = np.eye(3) * cross_variance
        process[3:, :3] = np.eye(3) * cross_variance
        process[3:, 3:] = np.eye(3) * velocity_variance
        return transition, process

    def _to_estimate(
        self,
        track: _Track,
        *,
        x: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        geometry_offset: np.ndarray | None = None,
        filter_timestamp: float | None = None,
    ) -> TrackEstimate:
        state_vector = track.x if x is None else np.asarray(x)
        state_covariance = (
            track.covariance if covariance is None else np.asarray(covariance)
        )
        offset = (
            np.zeros(3, dtype=np.float64)
            if geometry_offset is None
            else np.asarray(geometry_offset, dtype=np.float64).reshape(3)
        )
        aabb = track.aabb.translated(offset)
        obb = track.obb.translated(offset)
        source_points = (
            track.source_points.astype(np.float64) + offset
        ).astype(np.float32)
        nearest_point, nearest_distance = self._nearest_geometry(
            source_points,
            (
                track.nearest_point.astype(np.float64) + offset
            ).astype(np.float32),
        )
        predictions = self._future_positions(state_vector)
        elapsed = max(
            (track.filter_timestamp if filter_timestamp is None else filter_timestamp)
            - track.last_measurement_timestamp,
            0.0,
        )
        positional_uncertainty = float(
            np.sqrt(
                max(
                    float(
                        np.max(
                            np.linalg.eigvalsh(
                                state_covariance[:3, :3]
                            )
                        )
                    ),
                    0.0,
                )
            )
        )
        uncertainty_margin = (
            self.config.uncertainty_covariance_gain
            * positional_uncertainty
            + self.config.lost_uncertainty_gain_per_second * elapsed
        )
        return TrackEstimate(
            track_id=track.track_id,
            state=track.state,
            position=state_vector[:3].astype(np.float32).copy(),
            velocity=state_vector[3:].astype(np.float32).copy(),
            covariance=state_covariance.astype(np.float64).copy(),
            aabb=aabb,
            obb=obb,
            nearest_point=nearest_point,
            nearest_distance=nearest_distance,
            point_count=track.point_count,
            hit_count=track.hit_count,
            missed_count=track.missed_count,
            age_frames=track.age_frames,
            first_timestamp=track.first_timestamp,
            last_measurement_timestamp=track.last_measurement_timestamp,
            filter_timestamp=(
                track.filter_timestamp
                if filter_timestamp is None
                else float(filter_timestamp)
            ),
            confidence=float(np.clip(track.confidence, 0.0, 1.0)),
            pointcloud_quality=track.pointcloud_quality,
            mask_quality=track.mask_quality,
            mask_quality_score=track.mask_quality_score,
            source_points=source_points,
            source_colors=(
                None
                if track.source_colors is None
                else track.source_colors.copy()
            ),
            source_indices=(
                None
                if track.source_indices is None
                else track.source_indices.copy()
            ),
            predicted_positions=predictions,
            uncertainty_margin=float(max(uncertainty_margin, 0.0)),
            last_association_cost=track.last_association_cost,
        )

    def _future_positions(self, state_vector: np.ndarray) -> np.ndarray:
        if self.config.prediction_horizons_seconds is not None:
            times = np.asarray(
                self.config.prediction_horizons_seconds,
                dtype=np.float64,
            )
            return (
                state_vector[None, :3]
                + times[:, None] * state_vector[None, 3:]
            ).astype(np.float32)
        horizon = self.config.prediction_horizon_seconds
        step = self.config.prediction_step_seconds
        if horizon <= 0.0:
            return np.empty((0, 3), dtype=np.float32)
        times = np.arange(step, horizon + step * 0.5, step)
        return (
            state_vector[None, :3]
            + times[:, None] * state_vector[None, 3:]
        ).astype(np.float32)

    def _nearest_distance(self, point: np.ndarray) -> float:
        return float(
            np.linalg.norm(
                np.asarray(point, dtype=np.float32) - self._robot_origin
            )
        )

    def _nearest_geometry(
        self,
        points: np.ndarray,
        fallback_point: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        candidates = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(candidates):
            deltas = candidates - self._robot_origin
            index = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
            nearest_point = candidates[index].copy()
        else:
            nearest_point = np.asarray(
                fallback_point,
                dtype=np.float32,
            ).reshape(3).copy()
        return nearest_point, self._nearest_distance(nearest_point)
