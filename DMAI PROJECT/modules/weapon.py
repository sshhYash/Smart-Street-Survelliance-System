"""Weapon detection: crop each person, run the fine-tuned YOLO on the crops,
and confirm with temporal voting (e.g. 6 of the last 10 frames)."""
from collections import defaultdict, deque
from pathlib import Path

from ultralytics import YOLO

import config as C

FRAME_KEY = -999  # used when nobody is detected and we scan the whole frame


class WeaponDetector:
    def __init__(self, model_path=C.WEAPON_MODEL, device=None):
        self.device = device
        self.model = YOLO(str(model_path)) if Path(model_path).exists() else None
        self.history = defaultdict(lambda: deque(maxlen=C.WEAPON_WINDOW))
        if self.model is None:
            print(f"[weapon] {model_path} not found -> weapon module OFF "
                  f"(train it with scripts/train_weapon.py)")

    @property
    def enabled(self):
        return self.model is not None

    def detect(self, frame, persons):
        """persons: list of (track_id, box_xyxy).
        Returns (hits, confirmed): hits = {tid: [(box, conf), ...]} in frame
        coordinates; confirmed = list of tids that passed temporal voting."""
        if not self.enabled:
            return {}, []
        H, W = frame.shape[:2]
        crops, metas = [], []
        for tid, (x1, y1, x2, y2) in persons:
            mx, my = (x2 - x1) * C.WEAPON_CROP_MARGIN, (y2 - y1) * C.WEAPON_CROP_MARGIN
            cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            cx2, cy2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crops.append(frame[cy1:cy2, cx1:cx2])
            metas.append((tid, cx1, cy1))
        if not persons:  # nobody found: scan the full frame instead
            crops.append(frame)
            metas.append((FRAME_KEY, 0, 0))

        hits = {}
        if crops:
            results = self.model.predict(crops, imgsz=C.WEAPON_IMG_SIZE, conf=C.WEAPON_CONF,
                                         verbose=False, device=self.device)
            for (tid, ox, oy), r in zip(metas, results):
                found = []
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    found.append(((b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy), float(c)))
                if found:
                    hits[tid] = found

        confirmed = []
        keys = [tid for tid, _ in persons] or [FRAME_KEY]
        for tid in keys:
            self.history[tid].append(tid in hits)
            if sum(self.history[tid]) >= C.WEAPON_MIN_HITS:
                confirmed.append(tid)
        if len(self.history) > 500:  # forget people who left long ago
            for k in list(self.history)[:250]:
                del self.history[k]
        return hits, confirmed
