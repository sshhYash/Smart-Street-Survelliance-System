"""Train the 2-layer LSTM fight classifier on saved pose features.

  python scripts/train_fight.py

Saves models/fight_lstm.pt and results/fight_metrics.txt (clip-level accuracy, precision, recall, F1)."""
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from modules.fight import FEATURE_DIM, FightLSTM  # noqa: E402

SEQ = C.FIGHT_SEQ_LEN


def windows(feats, stride=8):
    if len(feats) < SEQ:
        pad = np.repeat(feats[:1], SEQ - len(feats), axis=0)
        feats = np.concatenate([pad, feats])
    return [feats[s:s + SEQ] for s in range(0, len(feats) - SEQ + 1, stride)]


def build(clips):
    X, y, owner = [], [], []
    for ci, c in enumerate(clips):
        for w in windows(c["feats"]):
            X.append(w)
            y.append(c["label"])
            owner.append(ci)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(owner)


def metrics(pred, true):
    tp = int(((pred == 1) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    acc = (tp + tn) / max(len(true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, f1, (tp, tn, fp, fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(C.DATA_DIR / "fight_features.pkl"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    torch.manual_seed(0)
    with open(a.data, "rb") as fh:
        clips = pickle.load(fh)
    random.Random(0).shuffle(clips)
    n_val = max(2, int(len(clips) * 0.2))
    val_clips, train_clips = clips[:n_val], clips[n_val:]

    Xtr, ytr, _ = build(train_clips)
    Xva, yva, own_va = build(val_clips)
    mean = Xtr.reshape(-1, FEATURE_DIM).mean(0)
    std = Xtr.reshape(-1, FEATURE_DIM).std(0)
    std[std < 1e-6] = 1.0
    Xtr, Xva = (Xtr - mean) / std, (Xva - mean) / std
    print(f"clips train/val: {len(train_clips)}/{len(val_clips)}  windows train/val: {len(Xtr)}/{len(Xva)}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FightLSTM().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t = torch.tensor(Xva, device=dev)

    def predict_val():
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(Xva_t)).cpu().numpy()

    best_acc, best_state = -1, None
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb, yb = Xtr_t[idx].to(dev), ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        acc = ((predict_val() > 0.5) == yva).mean()
        print(f"epoch {ep:2d}  loss {total / len(Xtr_t):.4f}  val window acc {acc * 100:.1f}%")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    probs = predict_val()
    clip_prob = np.array([probs[own_va == ci].mean() for ci in range(len(val_clips))])
    clip_true = np.array([c["label"] for c in val_clips])
    acc, prec, rec, f1, (tp, tn, fp, fn) = metrics((clip_prob > 0.5).astype(int), clip_true)

    C.MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state": best_state, "mean": mean.tolist(), "std": std.tolist(), "seq_len": SEQ,
                "feature_dim": FEATURE_DIM}, C.FIGHT_MODEL)
    text = (f"Fight recognition - pose + LSTM (validation clips: {len(val_clips)})\n"
            f"Accuracy  : {acc * 100:.1f}%\nPrecision : {prec * 100:.1f}%\n"
            f"Recall    : {rec * 100:.1f}%\nF1 score  : {f1 * 100:.1f}%\n"
            f"Confusion : TP={tp} TN={tn} FP={fp} FN={fn}\n"
            f"Best window accuracy: {best_acc * 100:.1f}%\n")
    print(text)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "fight_metrics.txt").write_text(text)
    print(f"Saved {C.FIGHT_MODEL}")


if __name__ == "__main__":
    main()
