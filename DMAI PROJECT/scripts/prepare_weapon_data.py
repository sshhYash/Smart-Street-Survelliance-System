"""Turn any downloaded weapon dataset (Kaggle / Roboflow, YOLO-style labels)
into a clean single-class 'weapon' dataset with an 80/20 train/val split.

  python scripts/prepare_weapon_data.py --src "C:/Downloads/weapons-dataset"

Handles: labels in a separate folder, multi-class labels (all become class 0),
polygon/segment labels (converted to boxes)."""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_line(line):
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(vals) == 4:
        x, y, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> box
        xs, ys = vals[0::2], vals[1::2]
        x, y = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        w, h = max(xs) - min(xs), max(ys) - min(ys)
    else:
        return None
    x, y = min(max(x, 0), 1), min(max(y, 0), 1)
    w, h = min(max(w, 0), 1), min(max(h, 0), 1)
    if w <= 0.001 or h <= 0.001:
        return None
    return f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def find_label(img, by_stem):
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    found = by_stem.get(img.stem)
    return found[0] if found and len(found) == 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder of the downloaded dataset")
    p.add_argument("--out", default=str(C.DATA_DIR / "weapons_yolo"))
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--max", type=int, default=0, help="use at most N images (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    src, out = Path(a.src), Path(a.out)
    images = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXT)
    by_stem = {}
    for t in src.rglob("*.txt"):
        by_stem.setdefault(t.stem, []).append(t)

    pairs, missing = [], 0
    for img in images:
        lbl = find_label(img, by_stem)
        if lbl is None:
            missing += 1
            continue
        lines = [l for l in (convert_line(x) for x in lbl.read_text(errors="ignore").splitlines()) if l]
        pairs.append((img, lines))
    if not pairs:
        sys.exit("No image/label pairs found. Check --src (it needs images + YOLO .txt labels).")

    random.Random(a.seed).shuffle(pairs)
    if a.max:
        pairs = pairs[:a.max]
    n_val = max(1, int(len(pairs) * a.val))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    if out.exists():
        shutil.rmtree(out)
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for i, (img, lines) in enumerate(items):
            name = f"{split}_{i:06d}"
            shutil.copy(img, out / "images" / split / f"{name}{img.suffix.lower()}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))

    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                    f"names:\n  0: weapon\n")
    boxes = sum(len(l) for _, l in pairs)
    print(f"Images found: {len(images)}  | without label (skipped): {missing}")
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Boxes: {boxes}")
    print(f"Wrote {yaml}")


if __name__ == "__main__":
    main()
