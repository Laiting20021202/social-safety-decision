from __future__ import annotations

from packages.common_models import Point2D, ZoneDefinition
from packages.overlay_renderer import ZoneStore


def test_zone_save_load_delete(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ZoneStore(tmp_path)
    zone = ZoneDefinition(
        zone_id="manual-demo",
        scenario_id="demo_crossing",
        name="Danger zone",
        source="manual",
        polygon=[
            Point2D(x=1, y=1),
            Point2D(x=100, y=1),
            Point2D(x=100, y=100),
        ],
        image_width=640,
        image_height=360,
    )

    path = store.save("michaelmunje/SocialNav-SUB", zone)
    loaded = store.load("michaelmunje/SocialNav-SUB", "demo_crossing")

    assert path.exists()
    assert loaded is not None
    assert loaded.zone_id == zone.zone_id
    assert store.delete("michaelmunje/SocialNav-SUB", "demo_crossing")
    assert store.load("michaelmunje/SocialNav-SUB", "demo_crossing") is None
