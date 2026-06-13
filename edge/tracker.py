import time
from collections import deque

import numpy as np

# GLOBALS
SIMILARITY_THRESHOLD = 0.75
PENDING_FRAMES = 2
EMBED_EVERY = 15
IDLE_THRESHOLD_PX = 30
IDLE_THRESHOLD_SEC = 10
AWAY_THRESHOLD_SEC = 5
PHONE_ASSIGN_PX = 200

class Person:

    def __init__(self, person_id: int, embedding: np.ndarray):

        # person tracking
        self.person_id = person_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.total_time = 0.0
        self.embedding = embedding
        self.embedding_count = 1

        # phone tracking
        self.phone_sightings = 0
        self.phone_time = 0.0
        self.phone_visible = False
        self.phone_start: float | None = None

        # idle tracking
        self.last_position: tuple[float, float] | None = None
        self.still_since: float | None = None
        self.idle_time = 0.0
        self.is_idle = False

        # away tracking
        self.away_time = 0.0
        self.away_start: float | None = None

    def add_embedding(self, embedding: np.ndarray):

        self.embedding_count += 1
        self.embedding += (embedding - self.embedding) / self.embedding_count
        self.embedding /= np.linalg.norm(self.embedding)

    def update_phone(self, phone_visible: bool, now: float):

        if phone_visible and not self.phone_visible:

            self.phone_sightings += 1
            self.phone_start = now
        elif not phone_visible and self.phone_visible and self.phone_start is not None:

            self.phone_time += now - self.phone_start
            self.phone_start = None
        self.phone_visible = phone_visible

    def update_idle(self, centroid: tuple[float, float], now: float):

        if self.last_position is None:
            self.last_position = centroid
            self.still_since = now
            return

        dist = np.hypot(centroid[0] - self.last_position[0], centroid[1] - self.last_position[1])
        if dist > IDLE_THRESHOLD_PX:

            if self.is_idle and self.still_since is not None:
                self.idle_time += now - self.still_since
            self.is_idle = False
            self.last_position = centroid
            self.still_since = now
        else:
            if self.still_since is not None and (now - self.still_since) >= IDLE_THRESHOLD_SEC:
                self.is_idle = True

    def mark_away(self, now: float):

        if self.away_start is None:
            self.away_start = self.last_seen

    def mark_returned(self, now: float):

        if self.away_start is not None:
            self.away_time += now - self.away_start
            self.away_start = None

        if self.phone_visible and self.phone_start is not None:
            self.phone_time += now - self.phone_start
            self.phone_start = None
            self.phone_visible = False


class Tracker:

    def __init__(self, extractor=None):

        if extractor is None:
            from reid import EmbeddingExtractor
            extractor = EmbeddingExtractor()
        self.extractor = extractor
        self.persons: dict[int, Person] = {}
        self.track_to_person: dict[int, int] = {}
        self.pending: dict[int, list[np.ndarray]] = {}
        self.next_person_id = 1
        self.frame_count = 0
        self._last_line_count = 0
        self._event_log: deque[str] = deque(maxlen=3)

    def update(self, person_results, phone_boxes, frame):

        self.frame_count += 1
        boxes = person_results[0].boxes
        now = time.time()

        if boxes.id is None:
            self._tick_away(set(), now)
            return

        track_ids = [int(i) for i in boxes.id.tolist()]
        coords = boxes.xyxy.cpu().numpy().astype(int)

        active_persons = {
            self.track_to_person[t] for t in track_ids if t in self.track_to_person
        }

        person_boxes_by_id: dict[int, np.ndarray] = {}

        for track_id, box in zip(track_ids, coords):
            if track_id in self.track_to_person:
                self._update_known(track_id, frame, box, now)
            else:
                self._identify_new(track_id, frame, box, active_persons, now)

            pid = self.track_to_person.get(track_id)
            if pid is not None:
                person_boxes_by_id[pid] = box

        self._tick_away(active_persons, now)
        self._update_phone(person_boxes_by_id, phone_boxes, now)
        self._update_idle(person_boxes_by_id, now)

    def _update_known(self, track_id, frame, box, now):

        person = self.persons[self.track_to_person[track_id]]
        person.total_time += now - person.last_seen
        person.last_seen = now
        if person.away_start is not None:
            person.mark_returned(now)

        if self.frame_count % EMBED_EVERY == 0:
            embedding = self.extractor.extract(self._crop(frame, box))
            if embedding is not None:
                person.add_embedding(embedding)

    def _identify_new(self, track_id, frame, box, active_persons, now):

        embedding = self.extractor.extract(self._crop(frame, box))
        if embedding is None:
            return

        samples = self.pending.setdefault(track_id, [])
        samples.append(embedding)
        if len(samples) < PENDING_FRAMES:
            return

        average = np.mean(samples, axis=0)
        average /= np.linalg.norm(average)
        del self.pending[track_id]

        person_id, similarity = self._best_match(average, exclude=active_persons)

        if person_id is not None:
            person = self.persons[person_id]
            person.mark_returned(now)
            person.last_seen = now
            person.add_embedding(average)
            self._event_log.append(
                f"  [re-id] person {person_id} returned (similarity {similarity:.2f})"
            )
        else:
            person_id = self.next_person_id
            self.next_person_id += 1
            self.persons[person_id] = Person(person_id, average)
            self._event_log.append(
                f"  [re-id] new person — ID {person_id}"
            )

        stale = [t for t, p in self.track_to_person.items() if p == person_id and t != track_id]
        for t in stale:
            del self.track_to_person[t]

        self.track_to_person[track_id] = person_id
        active_persons.add(person_id)

    def _tick_away(self, active_person_ids: set, now: float):

        for person in self.persons.values():
            if person.person_id not in active_person_ids:
                if (now - person.last_seen) > AWAY_THRESHOLD_SEC:
                    person.mark_away(now)

    def _update_phone(self, person_boxes: dict[int, np.ndarray], phone_boxes: np.ndarray, now: float):

        persons_with_phone: set[int] = set()

        for phone_box in phone_boxes:
            px = (phone_box[0] + phone_box[2]) / 2
            py = (phone_box[1] + phone_box[3]) / 2
            best_pid, best_dist = None, PHONE_ASSIGN_PX
            for pid, box in person_boxes.items():
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                dist = np.hypot(px - cx, py - cy)
                if dist < best_dist:
                    best_pid, best_dist = pid, dist
            if best_pid is not None:
                persons_with_phone.add(best_pid)

        for pid, person in self.persons.items():
            person.update_phone(pid in persons_with_phone, now)

    def _update_idle(self, person_boxes: dict[int, np.ndarray], now: float):

        for pid, box in person_boxes.items():
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            self.persons[pid].update_idle((cx, cy), now)

    def _best_match(self, embedding, exclude):

        best_id, best_similarity = None, 0.0
        for person_id, person in self.persons.items():
            if person_id in exclude:
                continue
            similarity = float(np.dot(embedding, person.embedding))
            if similarity > best_similarity:
                best_id, best_similarity = person_id, similarity

        if best_id is not None and best_similarity >= SIMILARITY_THRESHOLD:
            return best_id, best_similarity
        return None, best_similarity

    @staticmethod
    def _crop(frame, box):

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def get_stats(self, person_id: int) -> dict | None:

        person = self.persons.get(person_id)
        if person is None:
            return None
        phone_time = person.phone_time
        if person.phone_visible and person.phone_start:
            phone_time += time.time() - person.phone_start
        away_time = person.away_time
        if person.away_start:
            away_time += time.time() - person.away_start
        return {
            "person_id": person.person_id,
            "first_seen": person.first_seen,
            "last_seen": person.last_seen,
            "total_time_seconds": round(person.total_time, 2),
            "phone_sightings": person.phone_sightings,
            "phone_time_seconds": round(phone_time, 2),
            "idle_time_seconds": round(person.idle_time, 2),
            "is_idle": person.is_idle,
            "away_time_seconds": round(away_time, 2),
        }

    # matches the BGR COLORS list in main.py, converted to RGB for ANSI true color
    _ANSI_COLORS = [
        "\033[38;2;80;80;255m",    # blue
        "\033[38;2;255;80;80m",    # red
        "\033[38;2;80;255;80m",    # green
        "\033[38;2;0;180;255m",    # cyan
        "\033[38;2;255;0;180m",    # magenta
        "\033[38;2;255;200;0m",    # yellow
        "\033[38;2;0;140;255m",    # teal
        "\033[38;2;220;255;100m",  # lime
    ]
    _RESET = "\033[0m"

    def print_all(self):

        now = time.time()
        lines = []  # list of (text, person_id or None)

        for event in self._event_log:
            lines.append((event, None))
        if self._event_log:
            lines.append(("  " + "─" * 58, None))

        if not self.persons:
            lines.append(("  waiting for detections...", None))
        else:
            for person in self.persons.values():
                phone_time = person.phone_time
                if person.phone_visible and person.phone_start:
                    phone_time += now - person.phone_start
                away_time = person.away_time
                if person.away_start:
                    away_time += now - person.away_start

                status = "IDLE  " if person.is_idle else ("AWAY  " if person.away_start else "active")
                phone_str = f"  | 📱 {person.phone_sightings}x ({phone_time:.0f}s)" if person.phone_sightings else ""
                idle_str  = f"  | idle: {person.idle_time:.0f}s" if person.idle_time > 0 else ""
                away_str  = f"  | away: {away_time:.0f}s" if away_time > 0 else ""
                text = (f"  [{status}] person {person.person_id} | on floor: {person.total_time:.0f}s"
                        f"{phone_str}{idle_str}{away_str}")
                lines.append((text, person.person_id))

        # overwrite previous output
        if self._last_line_count:
            print(f"\033[{self._last_line_count}A", end="")

        for text, person_id in lines:
            padded = f"{text:<79}"
            if person_id is not None:
                color = self._ANSI_COLORS[(person_id - 1) % len(self._ANSI_COLORS)]
                print(f"{color}{padded}{self._RESET}")
            else:
                print(padded)

        self._last_line_count = len(lines)
