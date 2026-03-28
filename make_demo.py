"""
Make a demo video with action labels overlaid on OBS recording.

Usage:
    python make_demo.py --video path/to/obs_video.mp4 --session outputs/session_xxx

It will:
  1. Ask you to identify the sync marker (camera shake) time in the OBS video
  2. Align action timestamps with the video
  3. Cut 1 minute of video
  4. Overlay current action name on each frame
  5. Output: demo_xxx.mp4
"""

import argparse
import csv
import json
import pathlib
import sys
from datetime import datetime

import cv2
import numpy as np


def load_actions(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_wall_clock(s: str) -> float:
    """Parse ISO wall clock string to epoch seconds."""
    # Format: 2026-03-25T21:02:57.683
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
    return dt.timestamp()


def get_action_at_time(actions: list[dict], t: float) -> str | None:
    """Find which action is active at relative time t (seconds)."""
    for a in actions:
        start = float(a["start_time_sec"])
        end = float(a["end_time_sec"])
        if start <= t <= end + 0.1:  # small tolerance
            return a["action_name"]
    return None


def get_action_detail_at_time(actions: list[dict], t: float) -> tuple[str, str]:
    """Find action name and params at relative time t."""
    for a in actions:
        start = float(a["start_time_sec"])
        end = float(a["end_time_sec"])
        if start <= t <= end + 0.1:
            name = a["action_name"]
            params = a.get("params_json", "{}")
            return name, params
    return "", ""


def draw_label(frame: np.ndarray, action: str, params: str, t: float,
               font_scale: float = 1.2) -> np.ndarray:
    """Draw action label overlay on frame."""
    if not action:
        action = "(idle / transition)"

    h, w = frame.shape[:2]

    # Semi-transparent black bar at bottom
    bar_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Action name — large white text
    text = action
    cv2.putText(frame, text, (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Timestamp — smaller gray text
    time_text = f"t = {t:.1f}s"
    cv2.putText(frame, time_text, (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1,
                cv2.LINE_AA)

    # Params — right-aligned smaller text
    if params and params != "{}":
        # Shorten params for display
        p = params.replace('"', '').replace('{', '').replace('}', '')
        if len(p) > 60:
            p = p[:57] + "..."
        text_size = cv2.getTextSize(p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, p, (w - text_size[0] - 20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1,
                    cv2.LINE_AA)

    return frame


def _auto_detect_offset(video_path: str, actions: list[dict],
                        manual_offset: float | None) -> float:
    """Auto-detect the offset between OBS video start and session start.

    Logic:
      - OBS filename encodes recording start time (e.g. "2026-03-25 20-53-17.mp4")
      - actions[0].wall_clock_start is when the first action happened
      - offset = first_action_wall_clock - obs_start_time - first_action_session_time
    """
    if manual_offset is not None:
        print(f"Using manual sync offset: {manual_offset}s")
        return manual_offset

    # Try to parse OBS start time from filename
    video_name = pathlib.Path(video_path).stem  # e.g. "2026-03-25 20-53-17"
    obs_start = None
    for fmt in ["%Y-%m-%d %H-%M-%S", "%Y-%m-%d_%H-%M-%S", "%Y%m%d_%H%M%S"]:
        try:
            obs_start = datetime.strptime(video_name, fmt)
            break
        except ValueError:
            continue

    if obs_start is None:
        print(f"Warning: cannot parse OBS start time from filename '{video_name}'")
        print("Assuming video starts at session start (offset=0)")
        print("Use --sync-offset to manually specify")
        return 0.0

    # Get first action wall clock
    first_wc = actions[0]["wall_clock_start"]
    first_session_t = float(actions[0]["start_time_sec"])
    session_start = parse_wall_clock(first_wc) - first_session_t

    obs_epoch = obs_start.timestamp()
    offset = session_start - obs_epoch

    print(f"OBS recording started:  {obs_start.strftime('%H:%M:%S')}")
    print(f"Session started:        {datetime.fromtimestamp(session_start).strftime('%H:%M:%S.%f')[:-3]}")
    print(f"Auto-detected offset:   {offset:.1f}s")
    return offset


def main():
    parser = argparse.ArgumentParser(description="Make demo video with action overlays")
    parser.add_argument("--video", "-v", required=True, help="Path to OBS video file")
    parser.add_argument("--session", "-s", required=True,
                        help="Session directory (e.g. outputs/session_20260325_210252)")
    parser.add_argument("--sync-offset", type=float, default=None,
                        help="Manual sync offset: seconds in OBS video where sync_marker starts. "
                             "If not provided, uses 0 (assumes video starts at session start).")
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in session-relative seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=60,
                        help="Duration in seconds to cut (default: 60)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: demo_<session_id>.mp4)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for output video (e.g. 0.5 for half size)")
    args = parser.parse_args()

    # Load session data
    session_dir = pathlib.Path(args.session)
    csv_files = list(session_dir.glob("*_actions.csv"))
    json_files = list(session_dir.glob("*_summary.json"))
    if not csv_files or not json_files:
        print(f"Error: no actions.csv or summary.json in {session_dir}")
        sys.exit(1)

    actions = load_actions(str(csv_files[0]))
    summary = load_summary(str(json_files[0]))
    print(f"Loaded {len(actions)} actions from {csv_files[0].name}")

    # Auto-detect video offset from OBS filename + action wall clock
    # OBS filenames are like "2026-03-25 20-53-17.mp4"
    # The first action's wall_clock_start tells us when the session started
    video_offset = _auto_detect_offset(args.video, actions, args.sync_offset)
    print(f"Video offset: {video_offset:.1f}s (session actions start at this point in the video)")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    print(f"Video: {video_w}x{video_h} @ {video_fps:.1f}fps, {video_duration:.1f}s")

    # Calculate frame range
    # session time → video time: video_t = session_t + video_offset
    clip_start_session = args.start
    clip_end_session = args.start + args.duration

    clip_start_video = clip_start_session + video_offset
    clip_end_video = clip_end_session + video_offset

    start_frame = int(clip_start_video * video_fps)
    end_frame = int(clip_end_video * video_fps)

    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    print(f"Cutting: session {clip_start_session:.1f}s–{clip_end_session:.1f}s "
          f"→ video frames {start_frame}–{end_frame}")

    # Output
    out_w = int(video_w * args.scale)
    out_h = int(video_h * args.scale)

    out_path = args.output
    if not out_path:
        out_path = f"demo_{summary.get('session_id', 'output')}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (out_w, out_h))
    if not writer.isOpened():
        # Fallback to AVI
        out_path = out_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out_path, fourcc, video_fps, (out_w, out_h))

    print(f"Output: {out_path} ({out_w}x{out_h})")

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_num = start_frame
    processed = 0
    total_to_process = end_frame - start_frame

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Current time in session-relative seconds
        video_t = frame_num / video_fps
        session_t = video_t - video_offset

        # Find current action
        action_name, params = get_action_detail_at_time(actions, session_t)

        # Scale if needed
        if args.scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))

        # Draw overlay
        font_scale = 1.2 * args.scale if args.scale < 1.0 else 1.2
        frame = draw_label(frame, action_name, params, session_t,
                           font_scale=max(0.6, font_scale))

        writer.write(frame)
        frame_num += 1
        processed += 1

        if processed % 300 == 0:
            pct = processed / total_to_process * 100
            print(f"  {processed}/{total_to_process} frames ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"Done! Saved to {out_path}")
    print(f"  Frames: {processed}, Duration: {processed / video_fps:.1f}s")


if __name__ == "__main__":
    main()
