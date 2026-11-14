"""Alert engine: cooldown, snapshot, database row and optional Telegram message."""
import os
import re
import threading
import time
from datetime import datetime

import cv2

import config as C
from modules import db


class AlertEngine:
    def __init__(self, camera_id, camera_name):
        self.camera_id = camera_id
        self.camera_name = re.sub(r"[^A-Za-z0-9_-]", "_", camera_name)
        self.last_sent = {}
        self.token = os.environ.get("TELEGRAM_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        C.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        if self.token and self.chat_id:
            print("[alerts] Telegram ON")

    def raise_alert(self, alert_type, confidence, frame, message):
        """Returns True if a new alert was created (False during cooldown)."""
        now = time.time()
        if now - self.last_sent.get(alert_type, 0) < C.ALERT_COOLDOWN_SECONDS:
            return False
        self.last_sent[alert_type] = now
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = C.SNAPSHOT_DIR / f"{self.camera_name}_{alert_type}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        db.add_alert(self.camera_id, alert_type, round(float(confidence), 3), str(path), message)
        print(f"[ALERT] {datetime.now():%H:%M:%S} {self.camera_name} {alert_type.upper()}: {message}")
        if self.token and self.chat_id:
            text = f"{alert_type.upper()} alert - {self.camera_name}\n{message}"
            threading.Thread(target=self._send_telegram, args=(path, text), daemon=True).start()
        return True

    def _send_telegram(self, path, text):
        try:
            import requests
            with open(path, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{self.token}/sendPhoto",
                              data={"chat_id": self.chat_id, "caption": text},
                              files={"photo": f}, timeout=15)
        except Exception as e:  # never crash the video loop because of Telegram
            print(f"[alerts] Telegram failed: {e}")
