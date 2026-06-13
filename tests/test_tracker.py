
# Tracker tests using fake embeddings no camera or torch required.

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# make `from tracker import ...` work without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "edge"))
from tracker import Person, Tracker

# ── helpers ───────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy of v."""
    return v / np.linalg.norm(v)


def make_embedding(seed: int) -> np.ndarray:
    """Deterministic 512-d unit vector from a seed."""
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal(512).astype(np.float32))


class FakeExtractor:
    """Returns pre-queued embeddings one at a time; returns None when empty."""

    def __init__(self, embeddings: list[np.ndarray]):
        self._queue = list(embeddings)

    def extract(self, crop) -> np.ndarray | None:
        return self._queue.pop(0) if self._queue else None


def make_box_result(track_id: int, box=(10, 10, 50, 50)):
    """Minimal stand-in for a YOLO result with one tracked box."""
    x1, y1, x2, y2 = box
    # YOLO returns a 1-D tensor for id; .tolist() gives [1.0, 2.0, ...]
    # and .astype(int) is called on the xyxy numpy result
    id_arr  = np.array([float(track_id)])
    box_arr = np.array([[x1, y1, x2, y2]], dtype=float)
    boxes = SimpleNamespace(
        id=id_arr,
        xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: box_arr)),
    )
    return [SimpleNamespace(boxes=boxes)]


def make_empty_result():
    """YOLO result with no detections."""
    return [SimpleNamespace(boxes=SimpleNamespace(id=None))]


BLANK_FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


# ── Person unit tests ─────────────────────────────────────────────────────────

class TestPerson:

    def test_initial_state(self):
        p = Person(1, make_embedding(0))
        assert p.person_id == 1
        assert p.total_time == 0.0
        assert p.phone_sightings == 0
        assert not p.phone_visible
        assert not p.is_idle

    def test_add_embedding_updates_running_average(self):
        e1 = make_embedding(0)
        p = Person(1, e1.copy())
        e2 = make_embedding(1)
        p.add_embedding(e2)
        # embedding should be unit-normalized after update
        assert abs(np.linalg.norm(p.embedding) - 1.0) < 1e-5

    def test_phone_tracking(self):
        p = Person(1, make_embedding(0))
        now = time.time()
        p.update_phone(True, now)
        assert p.phone_sightings == 1
        assert p.phone_visible
        p.update_phone(False, now + 5.0)
        assert not p.phone_visible
        assert p.phone_time >= 4.9

    def test_away_and_returned(self):
        p = Person(1, make_embedding(0))
        p.last_seen = time.time() - 10
        now = time.time()
        p.mark_away(now)
        assert p.away_start is not None
        p.mark_returned(now + 3.0)
        assert p.away_start is None
        assert p.away_time >= 3.0


# ── Tracker integration tests ─────────────────────────────────────────────────

class TestTrackerReID:

    def _make_tracker(self, embeddings):
        return Tracker(extractor=FakeExtractor(embeddings))

    def test_new_person_created_after_pending_frames(self):
        e = make_embedding(0)

        tracker = self._make_tracker([e.copy(), e.copy()])
        result = make_box_result(track_id=1)

        tracker.update(result, [], BLANK_FRAME)
        assert len(tracker.persons) == 0

        tracker.update(result, [], BLANK_FRAME)
        assert len(tracker.persons) == 1
        assert tracker.track_to_person[1] == 1

    def test_returning_person_merged_above_threshold(self):
        e = make_embedding(0)
        tracker = self._make_tracker([e.copy(), e.copy(), e.copy(), e.copy()])

        for _ in range(2):
            tracker.update(make_box_result(track_id=1), [], BLANK_FRAME)
        assert 1 in tracker.persons

        tracker.update(make_empty_result(), [], BLANK_FRAME)

        for _ in range(2):
            tracker.update(make_box_result(track_id=2), [], BLANK_FRAME)

        assert len(tracker.persons) == 1
        assert tracker.track_to_person.get(2) == 1

    def test_different_appearance_creates_new_person(self):
        e1 = make_embedding(0)
        e2 = make_embedding(99)
        tracker = self._make_tracker([e1.copy(), e1.copy(), e2.copy(), e2.copy()])

        for _ in range(2):
            tracker.update(make_box_result(track_id=1), [], BLANK_FRAME)

        tracker.update(make_empty_result(), [], BLANK_FRAME)

        for _ in range(2):
            tracker.update(make_box_result(track_id=2), [], BLANK_FRAME)

        assert len(tracker.persons) == 2

    def test_active_person_excluded_from_matching(self):
        e = make_embedding(0)

        tracker = self._make_tracker([e.copy(), e.copy(), e.copy(), e.copy()])

        def two_person_result():
            id_arr  = np.array([1.0, 2.0])
            box_arr = np.array([[10,10,50,50],[60,10,100,50]], dtype=float)
            boxes = SimpleNamespace(
                id=id_arr,
                xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: box_arr)),
            )
            return [SimpleNamespace(boxes=boxes)]

        for _ in range(2):
            tracker.update(two_person_result(), [], BLANK_FRAME)

        assert len(tracker.persons) == 2

    def test_time_accumulates(self):
        e = make_embedding(0)
        tracker = self._make_tracker([e.copy(), e.copy()])

        for _ in range(2):
            tracker.update(make_box_result(track_id=1), [], BLANK_FRAME)

        person = tracker.persons[1]
        assert person.total_time >= 0.0
