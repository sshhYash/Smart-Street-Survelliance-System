"""Fine-tune YOLO11n on the weapon dataset and save models/weapon.pt.

  python scripts/train_weapon.py                       (defaults are fine for RTX 4060 8 GB)
  python scripts/train_weapon.py --epochs 25 --batch 8 (faster / less memory)"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(C.DATA_DIR / "weapons_yolo" / "data.yaml"))
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=10,
                workers=2, device=a.device, project=str(C.ROOT / "runs"), name="weapon",
                exist_ok=True, hsv_v=0.5, degrees=5)  # extra brightness change helps CCTV/night

    best = Path(model.trainer.best)
    C.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, C.WEAPON_MODEL)
    print(f"Saved {C.WEAPON_MODEL}")

    m = YOLO(str(C.WEAPON_MODEL)).val(data=a.data, imgsz=a.imgsz, device=a.device, plots=True)
    text = (f"Weapon detection (validation set)\n"
            f"mAP@50     : {m.box.map50 * 100:.1f}%\n"
            f"mAP@50-95  : {m.box.map * 100:.1f}%\n"
            f"Precision  : {m.box.mp * 100:.1f}%\n"
            f"Recall     : {m.box.mr * 100:.1f}%\n"
            f"Plots      : {m.save_dir}\n")
    print(text)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "weapon_metrics.txt").write_text(text)


if __name__ == "__main__":
    main()
