"""SQLite database: cameras, alerts, crowd_counts, users.
Video is never stored here - only events, numbers and snapshot file paths."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    location TEXT,
    rtsp_url TEXT,
    status TEXT DEFAULT 'offline',
    crowd_limit INTEGER DEFAULT 20
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL REFERENCES cameras(id),
    alert_type TEXT NOT NULL,
    confidence REAL,
    message TEXT,
    snapshot_path TEXT,
    created_at TEXT NOT NULL,
    reviewed INTEGER DEFAULT 0,
    is_false_alarm INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS crowd_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL REFERENCES cameras(id),
    people_count INTEGER NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'operator'
);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_counts_cam_time ON crowd_counts(camera_id, recorded_at);
"""


def now():
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_conn():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(C.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


def upsert_camera(name, location="", rtsp_url="", crowd_limit=C.CROWD_LIMIT):
    with get_conn() as c:
        row = c.execute("SELECT id FROM cameras WHERE name=?", (name,)).fetchone()
        if row:
            c.execute("UPDATE cameras SET location=?, rtsp_url=?, crowd_limit=? WHERE id=?",
                      (location, rtsp_url, crowd_limit, row["id"]))
            return row["id"]
        cur = c.execute("INSERT INTO cameras(name, location, rtsp_url, crowd_limit) VALUES (?,?,?,?)",
                        (name, location, rtsp_url, crowd_limit))
        return cur.lastrowid


def set_camera_status(camera_id, status):
    with get_conn() as c:
        c.execute("UPDATE cameras SET status=? WHERE id=?", (status, camera_id))


def add_alert(camera_id, alert_type, confidence, snapshot_path, message):
    with get_conn() as c:
        c.execute("INSERT INTO alerts(camera_id, alert_type, confidence, message, snapshot_path, created_at)"
                  " VALUES (?,?,?,?,?,?)",
                  (camera_id, alert_type, confidence, message, snapshot_path, now()))


def add_count(camera_id, people_count):
    with get_conn() as c:
        c.execute("INSERT INTO crowd_counts(camera_id, people_count, recorded_at) VALUES (?,?,?)",
                  (camera_id, int(people_count), now()))


def list_cameras():
    with get_conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM cameras ORDER BY name")]


def recent_alerts(camera_id=None, types=None, include_false=True, limit=30):
    q, args = "SELECT * FROM alerts WHERE 1=1", []
    if camera_id is not None:
        q += " AND camera_id=?"
        args.append(camera_id)
    if types:
        q += f" AND alert_type IN ({','.join('?' * len(types))})"
        args += list(types)
    if not include_false:
        q += " AND is_false_alarm=0"
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with get_conn() as c:
        return [dict(r) for r in c.execute(q, args)]


def counts_since(camera_id, minutes=10):
    since = (datetime.now() - timedelta(minutes=minutes)).isoformat(timespec="seconds")
    with get_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT people_count, recorded_at FROM crowd_counts WHERE camera_id=? AND recorded_at>=?"
            " ORDER BY id", (camera_id, since))]


def alerts_today(camera_id):
    today = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as c:
        return c.execute("SELECT COUNT(*) AS n FROM alerts WHERE camera_id=? AND created_at>=?",
                         (camera_id, today)).fetchone()["n"]


def review_alert(alert_id, false_alarm):
    with get_conn() as c:
        c.execute("UPDATE alerts SET reviewed=1, is_false_alarm=? WHERE id=?",
                  (1 if false_alarm else 0, alert_id))
