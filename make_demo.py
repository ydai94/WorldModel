"""
Make a demo video with action labels overlaid on OBS recording.

Usage:
    python make_demo.py --video path/to/obs_video.mp4 --session outputs/session_xxx

Reads the new events.csv format and overlays current WASD + camera state on each frame.
"""

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime

import cv2
import numpy as np


def load_events(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_wall_clock(s: str) -> float:
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return dt.timestamp()


def get_state_at_time(events: list[dict], t: float) -> dict:
    """Find the input state at session-relative time t.
    Returns the last event that occurred at or before time t."""
    state = {"w": 0, "a": 0, "s": 0, "d": 0, "mouse_dx": 0, "mouse_dy": 0, "event": ""}
    for e in events:
        et = float(e["timestamp_sec"])
        if et > t:
            break
        state = {
            "w": int(e["w"]),
            "a": int(e["a"]),
            "s": int(e["s"]),
            "d": int(e["d"]),
            "mouse_dx": int(e["mouse_dx"]),
            "mouse_dy": int(e["mouse_dy"]),
            "event": e["event"],
        }
    return state


def draw_overlay(frame: np.ndarray, state: dict, t: float,
                 font_scale: float = 1.0) -> np.ndarray:
    """Draw WASD state + camera info on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black bar at bottom
    bar_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # WASD indicator
    keys = []
    for k in ["w", "a", "s", "d"]:
        if state[k]:
            keys.append(k.upper())
    keys_text = " ".join(keys) if keys else "(no key)"

    # Camera indicator
    mdx, mdy = state["mouse_dx"], state["mouse_dy"]
    if mdx > 0:
        cam_text = "CAM: right"
    elif mdx < 0:
        cam_text = "CAM: left"
    elif mdy < 0:
        cam_text = "CAM: up"
    elif mdy > 0:
        cam_text = "CAM: down"
    else:
        cam_text = "CAM: --"

    # Draw keys — large
    cv2.putText(frame, f"KEYS: {keys_text}", (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Draw camera — large
    cv2.putText(frame, cam_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (150, 200, 255), 2,
                cv2.LINE_AA)

    # Timestamp — right side
    time_text = f"t = {t:.1f}s"
    ts_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
    cv2.putText(frame, time_text, (w - ts_size[0] - 20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1,
                cv2.LINE_AA)

    # Event label — right side
    event = state["event"]
    if event:
        ev_size = cv2.getTextSize(event, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, event, (w - ev_size[0] - 20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
                    cv2.LINE_AA)

    return frame


def _auto_detect_offset(video_path: str, events: list[dict],
                        manual_offset: float | None) -> float:
    if manual_offset is not None:
        print(f"Using manual sync offset: {manual_offset}s")
        return manual_offset

    video_name = pathlib.Path(video_path).stem
    obs_start = None
    for fmt in ["%Y-%m-%d %H-%M-%S", "%Y-%m-%d_%H-%M-%S", "%Y%m%d_%H%M%S"]:
        try:
            obs_start = datetime.strptime(video_name, fmt)
            break
        except ValueError:
            continue

    if obs_start is None:
        print(f"Warning: cannot parse OBS start time from '{video_name}'")
        print("Use --sync-offset to manually specify")
        return 0.0

    first_wc = events[0]["wall_clock"]
    first_t = float(events[0]["timestamp_sec"])
    session_start = parse_wall_clock(first_wc) - first_t

    obs_epoch = obs_start.timestamp()
    offset = session_start - obs_epoch

    print(f"OBS started:     {obs_start.strftime('%H:%M:%S')}")
    print(f"Session started: {datetime.fromtimestamp(session_start).strftime('%H:%M:%S.%f')[:-3]}")
    print(f"Offset:          {offset:.1f}s")
    return offset


def main():
    parser = argparse.ArgumentParser(description="Make demo video with event overlays")
    parser.add_argument("--video", "-v", required=True, help="OBS video file")
    parser.add_argument("--session", "-s", required=True, help="Session directory")
    parser.add_argument("--sync-offset", type=float, default=None)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    session_dir = pathlib.Path(args.session)

    # Find events CSV (try events first, fallback to actions for old format)
    csv_files = list(session_dir.glob("*_events.csv"))
    if not csv_files:
        csv_files = list(session_dir.glob("*_actions.csv"))
    json_files = list(session_dir.glob("*_summary.json"))
    if not csv_files or not json_files:
        print(f"Error: no events/actions csv or summary.json in {session_dir}")
        sys.exit(1)

    events = load_events(str(csv_files[0]))
    summary = load_summary(str(json_files[0]))
    print(f"Loaded {len(events)} events from {csv_files[0].name}")

    video_offset = _auto_detect_offset(args.video, events, args.sync_offset)
    print(f"Video offset: {video_offset:.1f}s")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_w}x{video_h} @ {video_fps:.1f}fps, {total_frames / video_fps:.1f}s")

    clip_start_video = args.start + video_offset
    clip_end_video = args.start + args.duration + video_offset

    start_frame = max(0, int(clip_start_video * video_fps))
    end_frame = min(total_frames, int(clip_end_video * video_fps))

    print(f"Cutting: session {args.start:.1f}s–{args.start + args.duration:.1f}s "
          f"→ frames {start_frame}–{end_frame}")

    out_w = int(video_w * args.scale)
    out_h = int(video_h * args.scale)

    # Make output width even (required by some codecs)
    out_w = out_w if out_w % 2 == 0 else out_w + 1
    out_h = out_h if out_h % 2 == 0 else out_h + 1

    out_path = args.output or f"demo_{summary.get('session_id', 'output')}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (out_w, out_h))
    if not writer.isOpened():
        out_path = out_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_path, fourcc, video_fps, (out_w, out_h))

    print(f"Output: {out_path} ({out_w}x{out_h})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame
    processed = 0
    total_to_process = end_frame - start_frame

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        session_t = frame_num / video_fps - video_offset
        state = get_state_at_time(events, session_t)

        if args.scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))

        fs = 1.0 * args.scale if args.scale < 1.0 else 1.0
        frame = draw_overlay(frame, state, session_t, font_scale=max(0.5, fs))

        writer.write(frame)
        frame_num += 1
        processed += 1

        if processed % 300 == 0:
            print(f"  {processed}/{total_to_process} ({processed / total_to_process * 100:.0f}%)")

    cap.release()
    writer.release()
    print(f"Done! {out_path} — {processed} frames, {processed / video_fps:.1f}s")


if __name__ == "__main__":
    main()
