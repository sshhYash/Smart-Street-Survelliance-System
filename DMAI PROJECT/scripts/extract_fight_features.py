"""Run YOLO11n-Pose on fight / normal clips and save per-frame pose features.

  python scripts/extract_fight_features.py ^
      --violence "D:/RLVS/Real Life Violence Dataset/Violence" ^
      --normal   "D:/RLVS/Real Life Violence Dataset/NonViolence" --max-per-class 200

Output: data/fight_features.pkl  (used by scripts/train_fight.py)"""
import argparse
import pickle
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from modules.fight import SpeedTracker, frame_features, people_from_result  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}


def video_features(path, device, max_frames):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    dt = 1.0 / fps if fps and 1 < fps < 120 else 1 / 25
    model = YOLO(C.POSE_MODEL)  # fresh model = fresh tracker for every clip
    speed = SpeedTracker()
    feats = []
    for i, r in enumerate(model.track(source=str(path), stream=True, persist=True, tracker="bytetrack.yaml",
                                      conf=C.PERSON_CONF, imgsz=C.IMG_SIZE, verbose=False, device=device)):
        if i >= max_frames:
            break
        people = people_from_result(r)
        feats.append(frame_features(people, speed.update(people, dt)))
    return np.stack(feats) if feats else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--violence", required=True, help="folder with fight clips")
    p.add_argument("--normal", required=True, help="folder with normal clips")
    p.add_argument("--max-per-class", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=150, help="frames used per clip")
    p.add_argument("--out", default=str(C.DATA_DIR / "fight_features.pkl"))
    p.add_argument("--device", default=None)
    a = p.parse_args()

    samples = []
    for label, folder in [(1, a.violence), (0, a.normal)]:
        vids = sorted(f for f in Path(folder).rglob("*") if f.suffix.lower() in VID_EXT)
        random.Random(0).shuffle(vids)
        vids = vids[:a.max_per_class]
        print(f"{'fight' if label else 'normal'}: {len(vids)} clips from {folder}")
        for v in tqdm(vids):
            try:
                f = video_features(v, a.device, a.max_frames)
            except Exception as e:
                print(f"  skip {v.name}: {e}")
                continue
            if f is not None and len(f) >= 8:
                samples.append({"feats": f, "label": label, "name": v.name})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(samples, fh)
    print(f"Saved {len(samples)} clips -> {a.out}")


if __name__ == "__main__":
    main()
