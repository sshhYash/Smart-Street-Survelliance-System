# Smart Street Surveillance System

Crowd estimation · Weapon detection · Fight recognition · Night (IR) mode
SQLite database · Streamlit dashboard · Telegram alerts (optional)

Ye code test kiya hua hai. Bina weapon/fight model ke bhi chalta hai:
weapon module OFF rahega aur fight **rule-based** chalega. Jaise hi
`models/weapon.pt` aur `models/fight_lstm.pt` ban jaate hain, wo apne aap use hone lagte hain.

---

## Folder

```
surveillance/
├── main.py                  ← poora system (video → detection → alerts)
├── dashboard.py             ← Streamlit dashboard
├── config.py                ← saari settings (limits, thresholds)
├── modules/                 ← video, night, crowd, weapon, fight, alerts, db
├── scripts/
│   ├── prepare_weapon_data.py   ← weapon dataset ko clean + split
│   ├── train_weapon.py          ← YOLO11n weapon training → models/weapon.pt
│   ├── extract_fight_features.py← fight clips se pose features
│   ├── train_fight.py           ← LSTM training → models/fight_lstm.pt
│   └── eval_crowd.py            ← ShanghaiTech pe crowd MAE
├── models/      (trained models yahan aate hain)
├── data/        (database, snapshots, datasets)
└── results/     (metrics .txt files — PPT ke numbers yahin se)
```

---

## Step 1: Setup (dono laptops pe, ~15 min)

Python 3.10, 3.11 ya 3.12 chahiye. Project folder mein terminal kholo:

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

GPU check karo, `True` aana chahiye:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
`False` aaye to pytorch.org → "Get Started" pe jaake apne system ka CUDA command copy karo.

---

## Step 2: System chalao (abhi, bina training ke)

Koi bhi street video `data/demo_videos/` mein rakho, phir:

```bash
python main.py --source data/demo_videos/street.mp4 --camera market_gate --crowd-limit 15
```

Doosre terminal mein dashboard:
```bash
streamlit run dashboard.py
```
Browser mein dashboard khulega: live feed, crowd graph aur alerts.

Aur options:
| Kya | Command |
|---|---|
| Laptop webcam | `python main.py --source 0` |
| Phone (IP Webcam app) | `python main.py --source http://PHONE_IP:8080/video` |
| CCTV / NVR | `python main.py --source rtsp://user:pass@IP:554/stream1` |
| Output video save | `--save output.mp4` (demo recording ke liye best) |
| Fight method | `--fight-mode rule` / `lstm` / `both` |

Window mein **Q** dabao to band.

---

## Step 3: Weapon model train karo (~1–1.5 ghanta GPU pe)

1. Kaggle se dataset download karo: **Weapons in Images** (`jubaerad/weapons-in-images-segmented-videos`) aur unzip karo.
   (Roboflow ka koi weapon dataset bhi chalega, YOLO format mein.)
2. Clean + split:
   ```bash
   python scripts/prepare_weapon_data.py --src "C:/Users/you/Downloads/weapons"
   ```
   Isse `data/weapons_yolo/data.yaml` ban jayega. Saari classes "weapon" ban jaati hain.
3. Train:
   ```bash
   python scripts/train_weapon.py
   ```
   "CUDA out of memory" aaye to `--batch 8` lagao. Jaldi chahiye to `--epochs 25`.
4. Result: `models/weapon.pt` + `results/weapon_metrics.txt` (mAP@50, precision, recall).
   Graphs (PR curve, confusion matrix) `runs/` folder mein milenge, PPT ke liye.

---

## Step 4: Fight LSTM train karo (~40 min)

1. Kaggle se **Real Life Violence Situations** download karo.
2. Pose features nikalo (200 + 200 clips):
   ```bash
   python scripts/extract_fight_features.py --violence "D:/RLVS/Real Life Violence Dataset/Violence" --normal "D:/RLVS/Real Life Violence Dataset/NonViolence" --max-per-class 200
   ```
3. LSTM train karo (~5 min):
   ```bash
   python scripts/train_fight.py
   ```
4. Result: `models/fight_lstm.pt` + `results/fight_metrics.txt` (accuracy, precision, recall, F1).

Agar apne demo video pe LSTM achha kaam na kare, to `--fight-mode rule` ya `both` use karo.

---

## Step 5: Crowd MAE (~5 min)

ShanghaiTech (Kaggle: `tthien/shanghaitech`) download karo:
```bash
python scripts/eval_crowd.py --data "D:/ShanghaiTech/part_B_final/test_data" --max 100
```
Result: `results/crowd_metrics.txt`. Bheed mein behtar count ke liye `--imgsz 1280` try karo.

---

## Telegram alerts (optional, 5 min)

1. Telegram pe **@BotFather** → `/newbot` → token copy karo.
2. Apne bot ko koi message bhejo, phir browser mein kholo:
   `https://api.telegram.org/bot<TOKEN>/getUpdates` → `"chat":{"id":...}` wala number copy karo.
3. `main.py` chalane se pehle (same terminal mein):
   ```bash
   set TELEGRAM_TOKEN=123456:ABC...        # PowerShell: $env:TELEGRAM_TOKEN="123456:ABC..."
   set TELEGRAM_CHAT_ID=987654321          # PowerShell: $env:TELEGRAM_CHAT_ID="987654321"
   ```

---

## Tuning (`config.py`)

| Problem | Kya badlo |
|---|---|
| Crowd alert jaldi/der se | `--crowd-limit` ya `CROWD_CONFIRM_SECONDS` |
| Weapon false alarm zyada | `WEAPON_CONF` badhao (0.55) ya `WEAPON_MIN_HITS` (7) |
| Weapon miss ho raha | `WEAPON_CONF` ghatao (0.35) |
| Fight alert normal walking pe | `FIGHT_WRIST_SPEED` badhao (2.0) |
| Fight detect nahi ho raha | `FIGHT_WRIST_SPEED` ghatao (1.0) ya `FIGHT_MIN_HITS` (6) |
| Slow (kam FPS) | `IMG_SIZE = 480`, ya `--save` hata do |

---

## Aaj ka plan (14 Nov)

| Time | Himanshu | Yash |
|---|---|---|
| 12:45–1:30 | Setup, `main.py` + dashboard ek video pe chalao | Setup, weapon dataset download |
| 1:30–3:00 | Demo videos record karo (bheed, acting fight, toy gun) | `prepare_weapon_data.py` → `train_weapon.py` start |
| 3:00–4:30 | Apne videos pe crowd + fight rule test, `config.py` tune | Training chalte waqt RLVS download → `extract_fight_features.py` |
| 4:30–5:30 | `eval_crowd.py` (ShanghaiTech) | `train_fight.py`, dono models Himanshu ko bhejo |
| 5:30–7:30 | Dono saath: models `models/` mein daalo, sab videos pe poora system chalao, `--save` se demo videos banao |
| 7:30–9:00 | `results/*.txt` se numbers PPT slide 13 mein bharo, dashboard ke screenshots lo |

**Demo safety:** acting wali fight aur khilona gun sirf room/campus mein, sabko bata ke record karo. Public jagah pe nahi.
