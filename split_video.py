"""
Split OBS video into per-session segments and convert to 720p.

Usage:
    python split_video.py --video ../WMVideos/2026-03-31\ 20-21-57.mp4 --sessions outputs/

It will:
  1. Find all session_* directories under --sessions
  2. Read each summary.json for wall_clock_start / wall_clock_end
  3. Parse OBS filename for recording start time
  4. Use ffmpeg to cut each segment and scale to 720p (preserving aspect ratio)
  5. Save each segment next to its session's actions.csv
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime


def get_ffmpeg_path() -> str:
    """Find ffmpeg: Mac uses conda/system PATH, Windows uses fixed path."""
    if sys.platform == "win32":
        win_path = r"C:\ffmpeg\bin\ffmpeg.exe"
        if pathlib.Path(win_path).exists():
            return win_path
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def parse_obs_start(video_path: str) -> datetime:
    """Parse OBS recording start time from filename like '2026-03-31 20-21-57.mp4'."""
    stem = pathlib.Path(video_path).stem
    for fmt in ["%Y-%m-%d %H-%M-%S", "%Y-%m-%d_%H-%M-%S", "%Y%m%d_%H%M%S"]:
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse OBS start time from filename: '{stem}'")


def parse_wall_clock(s: str) -> datetime:
    """Parse ISO wall clock string like '2026-03-31T20:27:37.508'."""
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")


def find_sessions(sessions_dir: str) -> list[pathlib.Path]:
    """Find all session directories with summary.json."""
    base = pathlib.Path(sessions_dir)
    sessions = sorted(base.glob("session_*/"))
    return [s for s in sessions if any(s.glob("*_summary.json"))]


def main():
    parser = argparse.ArgumentParser(description="Split OBS video into per-session 720p segments")
    parser.add_argument("--video", "-v", required=True, help="Path to OBS video file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sessions", "-s", default="outputs", help="Directory containing session_* folders")
    group.add_argument("--session", default=None, help="Single session directory")
    parser.add_argument("--height", type=int, default=720, help="Output height in pixels (default: 720)")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg CRF quality (default: 18, lower=better)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force", action="store_true", help="Overwrite existing 720p files")
    parser.add_argument("--obs-delay", type=float, default=1.0,
                        help="OBS startup delay in seconds (time between filename timestamp and actual first frame, default: 1.0)")
    args = parser.parse_args()

    # Find ffmpeg
    ffmpeg_bin = get_ffmpeg_path()
    print(f"Using ffmpeg: {ffmpeg_bin}")

    # Parse OBS start time
    try:
        obs_start = parse_obs_start(args.video)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"OBS recording start: {obs_start.strftime('%Y-%m-%d %H:%M:%S')}")

    # Find sessions
    if args.session:
        p = pathlib.Path(args.session)
        if any(p.glob("*_summary.json")):
            sessions = [p]
        else:
            print(f"No summary.json found in {args.session}")
            sys.exit(1)
    else:
        sessions = find_sessions(args.sessions)
    if not sessions:
        print(f"No sessions found")
        sys.exit(1)
    print(f"Found {len(sessions)} session(s)")

    obs_epoch = obs_start.timestamp()

    for session_dir in sessions:
        # Load summary
        summary_files = list(session_dir.glob("*_summary.json"))
        if not summary_files:
            continue
        with open(summary_files[0], "r", encoding="utf-8") as f:
            summary = json.load(f)

        session_id = summary.get("session_id", session_dir.name)
        wc_start = summary.get("wall_clock_start", "")
        wc_end = summary.get("wall_clock_end", "")

        if not wc_start or not wc_end:
            print(f"  [{session_id}] Skipping — missing wall_clock timestamps")
            continue

        # Calculate video offsets
        start_epoch = parse_wall_clock(wc_start).timestamp()
        end_epoch = parse_wall_clock(wc_end).timestamp()

        video_start = start_epoch - obs_epoch - args.obs_delay
        video_duration = end_epoch - start_epoch

        if video_start < 0:
            print(f"  [{session_id}] Skipping — session starts before OBS recording")
            continue

        # Output path — skip if already exists
        out_path = session_dir / f"session_{session_id}_720p.mp4"
        if out_path.exists() and not args.force:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  [{session_id}] Skipping — already exists ({size_mb:.1f} MB)")
            continue

        # Format timestamps for ffmpeg
        start_hms = f"{int(video_start // 3600):02d}:{int(video_start % 3600 // 60):02d}:{video_start % 60:06.3f}"
        dur_hms = f"{int(video_duration // 3600):02d}:{int(video_duration % 3600 // 60):02d}:{video_duration % 60:06.3f}"

        print(f"\n  [{session_id}]")
        print(f"    Video offset: {video_start:.1f}s — {video_start + video_duration:.1f}s")
        print(f"    Duration: {video_duration:.1f}s")
        print(f"    Output: {out_path}")

        # FFmpeg command
        # -ss before -i for fast keyframe seek, -ss after -i for precise trim
        # This avoids decoding from beginning (hours of data) while staying accurate
        safe_seek = max(0, video_start - 30)  # seek to 30s before target
        fine_offset = video_start - safe_seek  # precise trim within decoded window
        safe_hms = f"{int(safe_seek // 3600):02d}:{int(safe_seek % 3600 // 60):02d}:{safe_seek % 60:06.3f}"
        fine_hms = f"{int(fine_offset // 3600):02d}:{int(fine_offset % 3600 // 60):02d}:{fine_offset % 60:06.3f}"
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", safe_hms,
            "-i", args.video,
            "-ss", fine_hms,
            "-t", dur_hms,
            "-vf", f"scale=-2:{args.height}",
            "-vsync", "cfr",
            "-c:v", "libx264",
            "-crf", str(args.crf),
            "-c:a", "aac",
            "-preset", "fast",
            str(out_path),
        ]

        print(f"    CMD: {' '.join(cmd)}")

        if args.dry_run:
            print("    (dry run — skipped)")
            continue

        # Run ffmpeg with progress bar
        cmd_with_progress = cmd[:-1] + ["-progress", "pipe:1", str(out_path)]
        proc = subprocess.Popen(cmd_with_progress, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        bar_len = 40
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=")[1].strip())
                    elapsed_sec = us / 1_000_000
                    pct = min(elapsed_sec / video_duration, 1.0)
                    filled = int(bar_len * pct)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    print(f"\r    [{bar}] {pct * 100:.0f}%", end="", flush=True)
                except ValueError:
                    pass
        proc.wait()
        print()  # newline after progress bar
        if proc.returncode == 0:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"    Done — {size_mb:.1f} MB")
        else:
            err = proc.stderr.read()
            print(f"    FAILED: {err[-200:]}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
