import time

import numpy as np

from reid import EmbeddingExtractor

SIMILARITY_THRESHOLD = 0.80
PENDING_FRAMES = 5
EMBED_EVERY = 15


### Reps a single tracked person ###
class Person:

    def __init__(self, person_id: int, embedding: np.ndarray):
        self.person_id = person_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.total_time = 0.0
        self.embedding = embedding
        self.embedding_count = 1

    def add_embedding(self, embedding: np.ndarray):
        """Fold a new appearance sample into the running average."""
        self.embedding_count += 1
        self.embedding += (embedding - self.embedding) / self.embedding_count
        self.embedding /= np.linalg.norm(self.embedding)


### Records persons across frames, re-identifying them by appearance ###
class Tracker:

    def __init__(self):
        self.extractor = EmbeddingExtractor()
        self.persons: dict[int, Person] = {}
        self.track_to_person: dict[int, int] = {}
        self.pending: dict[int, list[np.ndarray]] = {}
        self.next_person_id = 1
        self.frame_count = 0

    def update(self, results, frame):
        self.frame_count += 1
        boxes = results[0].boxes

        if boxes.id is None:
            return

        track_ids = [int(i) for i in boxes.id.tolist()]
        coords = boxes.xyxy.cpu().numpy().astype(int)
        now = time.time()

        active_persons = {
            self.track_to_person[t] for t in track_ids if t in self.track_to_person
        }

        for track_id, box in zip(track_ids, coords):
            if track_id in self.track_to_person:
                self._update_known(track_id, frame, box, now)
            else:
                self._identify_new(track_id, frame, box, active_persons, now)

    def _update_known(self, track_id, frame, box, now):
        person = self.persons[self.track_to_person[track_id]]
        person.total_time += now - person.last_seen
        person.last_seen = now

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
            person.last_seen = now
            person.add_embedding(average)
            print(f"[tracker] person {person_id} returned "
                  f"(similarity {similarity:.2f}) — track {track_id}")
        else:
            person_id = self.next_person_id
            self.next_person_id += 1
            self.persons[person_id] = Person(person_id, average)
            print(f"[tracker] new person detected — ID {person_id} "
                  f"(best match was {similarity:.2f})")

        self.track_to_person[track_id] = person_id
        active_persons.add(person_id)

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
        return {
            "person_id": person.person_id,
            "first_seen": person.first_seen,
            "last_seen": person.last_seen,
            "total_time_seconds": round(person.total_time, 2),
        }

    def print_all(self):
        if not self.persons:
            print("[tracker] no persons tracked yet")
            return
        for person in self.persons.values():
            print(f"  ID {person.person_id} | on screen: {round(person.total_time, 2)}s")
