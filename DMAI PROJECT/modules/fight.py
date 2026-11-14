"""Fight recognition from body pose.

Two methods:
  rule  - two people close together + very fast wrist movement, confirmed over frames
  LSTM  - a 2-layer LSTM reads 32 frames of pose features for a pair of people
The same feature code is used for training (scripts/) and live inference."""
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config as C

N_KP = 17
L_WRIST, R_WRIST = 9, 10
FEATURE_DIM = N_KP * 2 * 2 + 3   # two people x 17 (x,y) + distance + 2 wrist speeds = 71
FAR_DIST = 5.0
MAX_SPEED = 10.0


# ---------- helpers shared by training and inference ----------
def people_from_result(result):
    """Ultralytics result -> list of (track_id, box_xyxy, keypoints[17,3])."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    if result.boxes.id is not None:
        ids = result.boxes.id.int().cpu().tolist()
    else:
        ids = [-(i + 1) for i in range(len(boxes))]
    kps = None
    if result.keypoints is not None and result.keypoints.data is not None:
        kps = result.keypoints.data.cpu().numpy()
        if kps.ndim == 3 and kps.shape[-1] == 2:
            kps = np.concatenate([kps, np.ones(kps.shape[:2] + (1,), kps.dtype)], -1)
    if kps is None or len(kps) != len(boxes):
        kps = np.zeros((len(boxes), N_KP, 3), np.float32)
    return [(int(t), b.astype(np.float32), k.astype(np.float32)) for t, b, k in zip(ids, boxes, kps)]


def box_h(box):
    return max(float(box[3] - box[1]), 1.0)


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float32)


def pair_distance(a, b):
    return float(np.linalg.norm(centre(a) - centre(b))) / ((box_h(a) + box_h(b)) / 2)


def wrist_speed(kp, prev_kp, box, dt):
    """Fastest wrist movement in body-heights per second."""
    if prev_kp is None or dt <= 0:
        return 0.0
    best = 0.0
    for j in (L_WRIST, R_WRIST):
        if kp[j, 2] > 0.5 and prev_kp[j, 2] > 0.5:
            best = max(best, float(np.linalg.norm(kp[j, :2] - prev_kp[j, :2])) / box_h(box) / dt)
    return min(best, MAX_SPEED)


def norm_keypoints(kp, box):
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    xy = (kp[:, :2] - np.array([x1, y1], np.float32)) / np.array([w, h], np.float32)
    xy[kp[:, 2] < 0.3] = 0.0
    return xy.flatten()


def pair_features(a, b, speeds):
    """a, b = (tid, box, kp); b may be None. The left person always comes first."""
    if b is not None and centre(b[1])[0] < centre(a[1])[0]:
        a, b = b, a
    fa, sa = norm_keypoints(a[2], a[1]), speeds.get(a[0], 0.0)
    if b is None:
        fb, sb, d = np.zeros(N_KP * 2, np.float32), 0.0, FAR_DIST
    else:
        fb, sb = norm_keypoints(b[2], b[1]), speeds.get(b[0], 0.0)
        d = min(pair_distance(a[1], b[1]), FAR_DIST)
    return np.concatenate([fa, fb, np.array([d, sa, sb], np.float32)]).astype(np.float32)


def frame_features(people, speeds):
    """Training features for one frame: the two largest people."""
    if not people:
        return np.zeros(FEATURE_DIM, np.float32)
    ranked = sorted(people, key=lambda p: (p[1][2] - p[1][0]) * (p[1][3] - p[1][1]), reverse=True)
    return pair_features(ranked[0], ranked[1] if len(ranked) > 1 else None, speeds)


class SpeedTracker:
    def __init__(self):
        self.prev = {}

    def update(self, people, dt):
        speeds = {tid: wrist_speed(kp, self.prev.get(tid), box, dt) for tid, box, kp in people}
        self.prev = {tid: kp for tid, _, kp in people}
        return speeds


class FightLSTM(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# ---------- live detector ----------
class FightDetector:
    def __init__(self, mode=C.FIGHT_MODE, model_path=C.FIGHT_MODEL, device=None):
        self.speed = SpeedTracker()
        self.rule_hist = deque(maxlen=C.FIGHT_WINDOW)
        self.lstm_hist = deque(maxlen=C.FIGHT_LSTM_CONFIRM)
        self.seqs = {}
        self.last_prob = 0.0
        self.lstm_pair = None
        self.frame_idx = 0
        self.model = None
        use_cuda = torch.cuda.is_available() and str(device).lower() != "cpu"
        self.device = torch.device("cuda" if use_cuda else "cpu")

        if mode != "rule" and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            self.model = FightLSTM().to(self.device)
            self.model.load_state_dict(ckpt["state"])
            self.model.eval()
            self.mean = torch.tensor(ckpt["mean"], dtype=torch.float32, device=self.device)
            self.std = torch.tensor(ckpt["std"], dtype=torch.float32, device=self.device)

        if mode == "auto":
            mode = "lstm" if self.model is not None else "rule"
        if mode in ("lstm", "both") and self.model is None:
            print(f"[fight] {model_path} not found -> using rule-based fight detection")
            mode = "rule"
        self.mode = mode
        print(f"[fight] method: {self.method}")

    @property
    def method(self):
        return {"lstm": "LSTM", "both": "LSTM + rule", "rule": "rule"}[self.mode]

    def update(self, people, dt):
        """people: list of (tid, box, kp). Returns (alarm, score, highlighted_pairs)."""
        self.frame_idx += 1
        speeds = self.speed.update(people, dt)
        by_id = {p[0]: p for p in people}

        # each person is paired with their nearest neighbour
        pairs = set()
        for a in people:
            best, best_d = None, 1e9
            for b in people:
                if b[0] == a[0]:
                    continue
                d = pair_distance(a[1], b[1])
                if d < best_d:
                    best, best_d = b, d
            if best is not None:
                pairs.add(tuple(sorted((a[0], best[0]))))

        rule_hit, hot_pairs = False, []
        for ta, tb in pairs:
            a, b = by_id[ta], by_id[tb]
            close = pair_distance(a[1], b[1]) < C.FIGHT_CLOSE_DIST
            fast = max(speeds[ta], speeds[tb]) > C.FIGHT_WRIST_SPEED
            if close and fast:
                rule_hit = True
                hot_pairs.append((ta, tb))
            if self.model is not None:
                seq = self.seqs.setdefault((ta, tb), deque(maxlen=C.FIGHT_SEQ_LEN))
                seq.append(pair_features(a, b, speeds))
        if self.model is not None:
            for k in list(self.seqs):
                if k not in pairs:
                    del self.seqs[k]

        self.rule_hist.append(rule_hit)
        rule_score = sum(self.rule_hist) / self.rule_hist.maxlen
        rule_alarm = sum(self.rule_hist) >= C.FIGHT_MIN_HITS

        lstm_alarm = False
        if self.model is not None:
            if self.frame_idx % C.FIGHT_LSTM_EVERY == 0:
                ready = [k for k, s in self.seqs.items() if len(s) == C.FIGHT_SEQ_LEN]
                prob = 0.0
                if ready:
                    x = torch.tensor(np.stack([np.stack(self.seqs[k]) for k in ready]),
                                     dtype=torch.float32, device=self.device)
                    x = (x - self.mean) / self.std
                    with torch.no_grad():
                        probs = torch.sigmoid(self.model(x)).cpu().numpy()
                    i = int(probs.argmax())
                    prob, self.lstm_pair = float(probs[i]), ready[i]
                self.last_prob = prob
                self.lstm_hist.append(prob >= C.FIGHT_LSTM_THRESHOLD)
            lstm_alarm = len(self.lstm_hist) == self.lstm_hist.maxlen and all(self.lstm_hist)

        if self.mode == "lstm":
            alarm, score = lstm_alarm, self.last_prob
        elif self.mode == "both":
            alarm, score = lstm_alarm or rule_alarm, max(self.last_prob, rule_score)
        else:
            alarm, score = rule_alarm, rule_score

        highlight = list(hot_pairs)
        if lstm_alarm and self.lstm_pair and all(t in by_id for t in self.lstm_pair):
            highlight.append(self.lstm_pair)
        return alarm, score, (highlight if alarm else [])
