"""Crowd counting error (MAE) on ShanghaiTech Part B test images.

  python scripts/eval_crowd.py --data "D:/ShanghaiTech/part_B/test_data" --max 100

The folder must contain images/ and ground-truth/ (GT_IMG_x.mat files)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def gt_count(mat_path):
    m = sio.loadmat(str(mat_path))
    return len(m["image_info"][0, 0][0, 0][0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="ShanghaiTech part_B test_data folder")
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=C.IMG_SIZE)
    p.add_argument("--conf", type=float, default=C.PERSON_CONF)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    root = Path(a.data)
    imgs = sorted((root / "images").glob("*.jpg"), key=lambda f: int(''.join(ch for ch in f.stem if ch.isdigit()) or 0))
    imgs = imgs[:a.max]
    model = YOLO(C.POSE_MODEL)
    errs, gts = [], []
    for img in imgs:
        gt = gt_count(root / "ground-truth" / f"GT_{img.stem}.mat")
        pred = len(model.predict(str(img), conf=a.conf, imgsz=a.imgsz, verbose=False, device=a.device)[0].boxes)
        errs.append(pred - gt)
        gts.append(gt)
    errs = np.array(errs)
    text = (f"Crowd counting on ShanghaiTech Part B ({len(imgs)} test images, imgsz {a.imgsz})\n"
            f"MAE  : {np.abs(errs).mean():.1f} people\n"
            f"RMSE : {np.sqrt((errs ** 2).mean()):.1f} people\n"
            f"Average true count: {np.mean(gts):.1f}\n")
    print(text)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "crowd_metrics.txt").write_text(text)


if __name__ == "__main__":
    main()
