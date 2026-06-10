import time 


### Reps a single tracked person ###
class Person:

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.first_seen = time.time()   # first seen timestamp
        self.last_seen = time.time()    # last seen timestamp
        self.total_time = 0.0           # total time seen



### Record persons details across frames ###
class tracker:

    def __init__(self):
        self.persons: dict[int, Person] = {}

    def update(self, results):
        boxes = results[0].boxes

        if boxes.id is None:
            return

        track_ids = [int(i) for i in boxes.id.tolist()]
        now = time.time()

        for track_id in track_ids:
            if track_id not in self.persons:
                self.persons[track_id] = Person(track_id)
                print(f"[tracker] new person detected — ID {track_id}")
            else:
                person = self.persons[track_id]
                person.total_time += now - person.last_seen
                person.last_seen = now

    def get_stats(self, track_id: int) -> dict | None:
        person = self.persons.get(track_id)
        if person is None:
            return None
        return {
            "track_id": person.track_id,
            "first_seen": person.first_seen,
            "last_seen": person.last_seen,
            "total_time_seconds": round(person.total_time, 2),
        }

    def print_all(self):
        if not self.persons:
            print("[tracker] no persons tracked yet")
            return
        for person in self.persons.values():
            print(f"  ID {person.track_id} | on screen: {round(person.total_time, 2)}s")

