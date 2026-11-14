"""All settings in one place. Change numbers here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
LIVE_DIR = DATA_DIR / "live"
RESULTS_DIR = ROOT / "results"
DB_PATH = DATA_DIR / "surveillance.db"
MODELS_DIR = ROOT / "models"

# Models
POSE_MODEL = "yolo11n-pose.pt"          # person boxes + 17 keypoints (auto-downloads)
WEAPON_MODEL = MODELS_DIR / "weapon.pt"   # created by scripts/train_weapon.py
FIGHT_MODEL = MODELS_DIR / "fight_lstm.pt"  # created by scripts/train_fight.py

# Person detection
PERSON_CONF = 0.35
IMG_SIZE = 640
MAX_FRAME_WIDTH = 1280

# Crowd
CROWD_LIMIT = 20               # default limit, override with --crowd-limit
CROWD_SMOOTH_SECONDS = 3       # moving average window
CROWD_CONFIRM_SECONDS = 3      # count must stay above limit this long

# Weapon
WEAPON_CONF = 0.45
WEAPON_CROP_MARGIN = 0.25      # extra space around each person box
WEAPON_IMG_SIZE = 416          # crops are resized (upscaled) to this
WEAPON_WINDOW = 10             # temporal voting: last N frames
WEAPON_MIN_HITS = 6            # ...weapon must be seen in this many

# Fight
FIGHT_MODE = "auto"            # auto = LSTM if trained model exists, else rule
FIGHT_WRIST_SPEED = 1.5        # body-heights per second (rule)
FIGHT_CLOSE_DIST = 1.2         # centre distance / body height (rule)
FIGHT_WINDOW = 20              # rule voting window (frames)
FIGHT_MIN_HITS = 10            # rule hits needed inside the window
FIGHT_SEQ_LEN = 32             # LSTM sequence length (frames)
FIGHT_LSTM_EVERY = 8           # run LSTM every N frames
FIGHT_LSTM_THRESHOLD = 0.7
FIGHT_LSTM_CONFIRM = 3         # consecutive LSTM windows above threshold

# Night / low light
IR_COLOUR_THRESHOLD = 0.15     # colourfulness (relative to brightness) below this = IR
LOW_LIGHT_THRESHOLD = 60       # mean brightness below this = low light

# Alerts
ALERT_COOLDOWN_SECONDS = 30    # per camera, per alert type
LIVE_FRAME_EVERY = 3           # write dashboard frame every N frames
