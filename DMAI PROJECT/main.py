"""Smart Street Surveillance System - main pipeline.

Examples:
  python main.py --source data/demo_videos/street.mp4 --camera market_gate --crowd-limit 15
  python main.py --source rtsp://user:pass@192.168.1.20:554/stream1 --camera cam1
  python main.py --source 0                       (laptop webcam)
Press Q in the video window to stop."""
import argparse
import os
import re
import time
from collections import deque

import cv2
from ultralytics import YOLO

import config as C
from modules import db
from modules.alerts import AlertEngine
from modules.crowd import CrowdCounter
from modules.fight import FightDetector, people_from_result
from modules.night import enhance, scene_mode
from modules.video import VideoSource
from modules.weapon import FRAME_KEY, WeaponDetector

BLUE, ORANGE, RED = (235, 120, 30), (0, 140, 255), (60, 60, 230)
WHITE, DARK = (245, 245, 245), (43, 32, 21)


def parse_args():
    p = argparse.ArgumentParser(description="Smart Street Surveillance System")
    p.add_argument("--source", required=True, help="video file, RTSP URL, or webcam index (0)")
    p.add_argument("--camera", default="cam1", help="camera name shown on the dashboard")
    p.add_argument("--location", default="")
    p.add_argument("--crowd-limit", type=int, default=C.CROWD_LIMIT)
    p.add_argument("--fight-mode", choices=["auto", "rule", "lstm", "both"], default=C.FIGHT_MODE)
    p.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (default: auto)")
    p.add_argument("--save", default="", help="optional path to save the annotated video (.mp4)")
    p.add_argument("--no-show", action="store_true", help="do not open a video window")
    return p.parse_args()


def label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw(frame, people, weapon_hits, weapon_confirmed, fight_pairs, header, banner):
    vis = frame.copy()
    fighting = {t for pair in fight_pairs for t in pair}
    for tid, box, _ in people:
        x1, y1, x2, y2 = map(int, box)
        color = RED if tid in fighting else ORANGE if tid in weapon_confirmed else BLUE
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label(vis, f"ID {tid}" if tid >= 0 else "person", x1, y1, color)
    for hits in weapon_hits.values():
        for (bx1, by1, bx2, by2), conf in hits:
            cv2.rectangle(vis, (int(bx1), int(by1)), (int(bx2), int(by2)), ORANGE, 2)
            label(vis, f"weapon {conf:.2f}", int(bx1), int(by1), ORANGE)
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 34), DARK, -1)
    cv2.putText(vis, header, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    if banner:
        cv2.rectangle(vis, (0, h - 40), (w, h), RED, -1)
        cv2.putText(vis, banner, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    return vis


def write_live_frame(img, camera_name):
    C.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    final = C.LIVE_DIR / f"{camera_name}.jpg"
    tmp = C.LIVE_DIR / f"{camera_name}.tmp.jpg"
    cv2.imwrite(str(tmp), img)
    try:
        os.replace(tmp, final)
    except OSError:
        pass  # dashboard was reading the file; next frame will succeed


def main():
    args = parse_args()
    cam_name = re.sub(r"[^A-Za-z0-9_-]", "_", args.camera)
    src = VideoSource(args.source)
    proc_fps = src.fps if src.is_file else 15.0

    print("[init] loading models ...")
    pose = YOLO(C.POSE_MODEL)
    weapon = WeaponDetector(device=args.device)
    fight = FightDetector(mode=args.fight_mode, device=args.device)
    crowd = CrowdCounter(args.crowd_limit, proc_fps, C.CROWD_SMOOTH_SECONDS, C.CROWD_CONFIRM_SECONDS)

    db.init_db()
    cam_id = db.upsert_camera(cam_name, args.location, str(args.source), args.crowd_limit)
    db.set_camera_status(cam_id, "online")
    alerts = AlertEngine(cam_id, cam_name)

    writer = None
    frame_no, last_count_save, t_prev = 0, 0.0, time.time()
    loop_times = deque(maxlen=30)
    banner, banner_until = "", 0.0
    print("[run] press Q in the window to stop")
    try:
        while True:
            frame = src.read()
            if frame is None:
                print("[run] end of video")
                break
            t0 = time.time()
            frame_no += 1
            if frame.shape[1] > C.MAX_FRAME_WIDTH:
                scale = C.MAX_FRAME_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            # 1. day / night
            mode = scene_mode(frame)
            proc = enhance(frame, mode)

            # 2. people + tracking + pose (one model)
            res = pose.track(proc, persist=True, tracker="bytetrack.yaml", conf=C.PERSON_CONF,
                             imgsz=C.IMG_SIZE, verbose=False, device=args.device)[0]
            people = people_from_result(res)
            dt = 1.0 / src.fps if src.is_file else max(t0 - t_prev, 1e-3)
            t_prev = t0

            # 3. modules
            count, crowd_alarm = crowd.update(len(people))
            w_hits, w_confirmed = weapon.detect(proc, [(t, b) for t, b, _ in people])
            fight_alarm, fight_score, fight_pairs = fight.update(people, dt)

            # 4. draw
            loop_times.append(time.time() - t0)
            fps = len(loop_times) / max(sum(loop_times), 1e-6)
            header = (f"{cam_name} | {mode} | People: {count}/{args.crowd_limit} | "
                      f"Weapon: {'ON' if weapon.enabled else 'OFF'} | Fight: {fight.method} | {fps:.1f} FPS")
            active = []
            if crowd_alarm:
                active.append(f"OVERCROWDING ({count})")
            if w_confirmed:
                active.append("WEAPON")
            if fight_alarm:
                active.append("FIGHT")
            if active:
                banner, banner_until = "ALERT: " + " + ".join(active), time.time() + 3
            vis = draw(proc, people, w_hits, w_confirmed, fight_pairs, header,
                       banner if time.time() < banner_until else "")

            # 5. alerts (snapshot includes the boxes)
            if crowd_alarm:
                alerts.raise_alert("crowd", count, vis,
                                   f"Overcrowding: {count} people (limit {args.crowd_limit})")
            if w_confirmed:
                conf = max((c for t in w_confirmed for _, c in w_hits.get(t, [])), default=0.0)
                who = ", ".join(f"ID {t}" for t in w_confirmed if t != FRAME_KEY) or "frame"
                alerts.raise_alert("weapon", conf, vis, f"Weapon seen with {who}")
            if fight_alarm:
                who = ", ".join(f"ID {a} & ID {b}" for a, b in fight_pairs) or "people in view"
                alerts.raise_alert("fight", fight_score, vis, f"Possible fight: {who} ({fight.method})")

            # 6. database + dashboard frame
            if time.time() - last_count_save >= 1.0:
                db.add_count(cam_id, count)
                last_count_save = time.time()
            if frame_no % C.LIVE_FRAME_EVERY == 0:
                write_live_frame(vis, cam_name)

            if args.save:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), proc_fps, (w, h))
                writer.write(vis)
            if not args.no_show:
                cv2.imshow("Smart Street Surveillance", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        db.set_camera_status(cam_id, "offline")
        src.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if loop_times:
            print(f"[done] average speed: {len(loop_times) / sum(loop_times):.1f} FPS (last 30 frames)")


if __name__ == "__main__":
    main()
