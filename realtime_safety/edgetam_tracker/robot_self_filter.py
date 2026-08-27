from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from realtime_safety.edgetam_tracker.models import CloudFrame


class SelfFilterStatus(str, Enum):
    DISABLED = "DISABLED"
    UNAVAILABLE = "UNAVAILABLE"
    ACTIVE = "ACTIVE"


@dataclass(slots=True)
class RobotSelfFilterConfig:
    enabled: bool = False
    padding: float = 0.02
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.padding < 0:
            raise ValueError("self-filter padding cannot be negative")


SelfFilterConfig = RobotSelfFilterConfig


@dataclass(slots=True)
class LinkSphere:
    center: np.ndarray
    radius: float
    link_name: str = ""

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        self.radius = float(self.radius)
        if not np.isfinite(self.center).all():
            raise ValueError("link sphere center must be finite")
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("link sphere radius must be finite and non-negative")


@dataclass(slots=True)
class SelfFilterResult:
    cloud: CloudFrame
    status: SelfFilterStatus
    removed_points: int
    input_points: int
    sphere_count: int
    fail_closed: bool
    reason: str = ""

    @property
    def filtered_cloud(self) -> CloudFrame:
        return self.cloud

    @property
    def available(self) -> bool:
        return self.status is SelfFilterStatus.ACTIVE

    @property
    def safe_to_continue(self) -> bool:
        """Whether a caller may continue without a fail-closed diagnostic."""

        return self.status is not SelfFilterStatus.UNAVAILABLE or not self.fail_closed


class RobotSelfFilter:
    """Remove points inside already-transformed robot-link spheres.

    TF lookup and URDF parsing deliberately stay in the ROS adapter.  This core
    accepts sphere centers expressed in ``cloud.frame_id`` so it cannot silently
    mix frames.  When geometry is unavailable it preserves every point; clearing
    the cloud would incorrectly make an obstacle disappear.  ``safe_to_continue``
    tells a fail-closed ROS node to raise a degraded/stop diagnostic instead.
    """

    def __init__(
        self,
        config: RobotSelfFilterConfig | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = RobotSelfFilterConfig(**overrides)
        elif overrides:
            config = replace(config, **overrides)
        self.config = config

    def filter(
        self,
        cloud: CloudFrame,
        link_spheres: (
            Sequence[LinkSphere | Mapping[str, Any] | tuple[Any, ...]]
            | Mapping[str, LinkSphere | Mapping[str, Any] | tuple[Any, ...]]
            | None
        ) = None,
    ) -> SelfFilterResult:
        if not self.config.enabled:
            return SelfFilterResult(
                cloud=cloud,
                status=SelfFilterStatus.DISABLED,
                removed_points=0,
                input_points=len(cloud.points),
                sphere_count=0,
                fail_closed=self.config.fail_closed,
                reason="robot self-filter disabled",
            )

        try:
            spheres = self._normalize_spheres(link_spheres)
        except (TypeError, ValueError) as exc:
            return SelfFilterResult(
                cloud=cloud,
                status=SelfFilterStatus.UNAVAILABLE,
                removed_points=0,
                input_points=len(cloud.points),
                sphere_count=0,
                fail_closed=self.config.fail_closed,
                reason=f"invalid robot link geometry: {exc}",
            )
        if not spheres:
            return SelfFilterResult(
                cloud=cloud,
                status=SelfFilterStatus.UNAVAILABLE,
                removed_points=0,
                input_points=len(cloud.points),
                sphere_count=0,
                fail_closed=self.config.fail_closed,
                reason="no transformed robot link spheres available",
            )

        inside = np.zeros(len(cloud.points), dtype=bool)
        points = np.asarray(cloud.points, dtype=np.float32)
        for sphere in spheres:
            radius = float(sphere.radius + self.config.padding)
            squared_distance = np.einsum(
                "ij,ij->i",
                points - sphere.center,
                points - sphere.center,
            )
            inside |= squared_distance <= radius * radius
        filtered = cloud.select(~inside)
        return SelfFilterResult(
            cloud=filtered,
            status=SelfFilterStatus.ACTIVE,
            removed_points=int(np.count_nonzero(inside)),
            input_points=len(cloud.points),
            sphere_count=len(spheres),
            fail_closed=self.config.fail_closed,
        )

    process = filter

    @staticmethod
    def _normalize_spheres(
        values: (
            Sequence[LinkSphere | Mapping[str, Any] | tuple[Any, ...]]
            | Mapping[str, LinkSphere | Mapping[str, Any] | tuple[Any, ...]]
            | None
        ),
    ) -> list[LinkSphere]:
        if values is None:
            return []
        if isinstance(values, Mapping):
            items = list(values.items())
        else:
            items = [("", value) for value in values]

        result: list[LinkSphere] = []
        for default_name, raw in items:
            if isinstance(raw, LinkSphere):
                result.append(raw)
                continue
            if isinstance(raw, Mapping):
                if "center" not in raw or "radius" not in raw:
                    raise ValueError("sphere mappings require center and radius")
                result.append(
                    LinkSphere(
                        center=raw["center"],
                        radius=raw["radius"],
                        link_name=str(raw.get("link_name", default_name)),
                    )
                )
                continue
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise TypeError(f"unsupported sphere description: {type(raw).__name__}")
            if len(raw) == 2:
                center, radius = raw
                name = default_name
            elif len(raw) == 3:
                name, center, radius = raw
            else:
                raise ValueError("sphere tuples must be (center, radius) or (name, center, radius)")
            result.append(LinkSphere(center=center, radius=radius, link_name=str(name)))
        return result


__all__ = [
    "LinkSphere",
    "RobotSelfFilter",
    "RobotSelfFilterConfig",
    "SelfFilterConfig",
    "SelfFilterResult",
    "SelfFilterStatus",
]
