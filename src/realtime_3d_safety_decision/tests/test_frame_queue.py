from realtime_safety.pipeline.frame_queue import LatestQueue


def test_queue_never_exceeds_capacity_and_drops_oldest() -> None:
    queue = LatestQueue[int](maxsize=2)
    queue.put_latest(1)
    queue.put_latest(2)
    queue.put_latest(3)
    assert queue.qsize() == 2
    assert queue.dropped == 1
    assert queue.get(timeout=0.01) == 2
    assert queue.get(timeout=0.01) == 3


def test_get_latest_discards_stale_items() -> None:
    queue = LatestQueue[int](maxsize=3)
    for value in range(3):
        queue.put_latest(value)
    assert queue.get_latest(timeout=0.01) == 2
    assert queue.empty()
    assert queue.dropped == 2
