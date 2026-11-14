"""Crowd count = tracked people, smoothed with a moving average."""
from collections import deque


class CrowdCounter:
    def __init__(self, limit, fps, smooth_seconds, confirm_seconds):
        self.limit = limit
        self.history = deque(maxlen=max(1, int(fps * smooth_seconds)))
        self.confirm_needed = max(1, int(fps * confirm_seconds))
        self.frames_over = 0

    def update(self, n_people):
        """Return (smoothed_count, alarm)."""
        self.history.append(n_people)
        avg = sum(self.history) / len(self.history)
        self.frames_over = self.frames_over + 1 if avg > self.limit else 0
        return int(round(avg)), self.frames_over >= self.confirm_needed
