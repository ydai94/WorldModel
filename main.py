"""
Cyberpunk 2077 — World Model Training Data Collector
=====================================================
Automated first-person exploration with concurrent random movement and camera.
Video recording handled externally by OBS Studio.
Outputs event-driven logs with wall-clock timestamps for OBS alignment.

Target platform: Windows 10/11, Python 3.10+
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import pathlib
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Platform guard
# ---------------------------------------------------------------------------
_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    import pydirectinput
    import keyboard

    pydirectinput.PAUSE = 0.0
else:
    pydirectinput = None  # type: ignore
    keyboard = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cp2077_collector")


# ===================================================================
# 1. Configuration
# ===================================================================

@dataclass
class Config:
    game_name: str = "Cyberpunk2077"
    duration_sec: float = 600.0
    output_dir: str = "outputs"

    # Movement
    enable_random_movement: bool = True
    min_move_duration_sec: float = 0.5
    max_move_duration_sec: float = 3.0
    idle_probability: float = 0.1

    # Camera observation
    enable_camera_scan: bool = True
    scan_candidates: list[str] = field(
        default_factory=lambda: ["right_90", "left_90", "look_up", "look_down", "hold"]
    )
    scan_weights: list[float] = field(
        default_factory=lambda: [0.20, 0.20, 0.20, 0.20, 0.20]
    )
    scan_duration_sec: float = 2.0
    scan_hold_duration_sec: float = 1.5
    mouse_sensitivity: float = 10.0
    mouse_sensitivity_y: float = 10.0

    # Probability of doing camera + movement at the same time
    concurrent_probability: float = 0.5

    # Hotkeys
    hotkey_stop: str = "F8"
    hotkey_pause: str = "F9"

    # Countdown
    countdown_sec: int = 5

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> Config:
        p = pathlib.Path(path)
        if not p.exists():
            log.warning("Config %s not found — using defaults.", p)
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {fld.name for fld in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ===================================================================
# 2. Event Logger
# ===================================================================

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


class EventLogger:
    """Event-driven logger: one row per state change.
    Tracks which keys are held and current mouse movement rate."""

    _FIELDS = ["timestamp_sec", "wall_clock", "w", "a", "s", "d",
               "mouse_dx", "mouse_dy", "event"]

    def __init__(self, t0: float) -> None:
        self._t0 = t0
        self._records: list[dict] = []
        self._keys = {"w": 0, "a": 0, "s": 0, "d": 0}
        self._mouse_dx = 0
        self._mouse_dy = 0

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _snapshot(self, event: str) -> dict:
        return {
            "timestamp_sec": round(self._elapsed(), 4),
            "wall_clock": _now_iso(),
            "w": self._keys["w"],
            "a": self._keys["a"],
            "s": self._keys["s"],
            "d": self._keys["d"],
            "mouse_dx": self._mouse_dx,
            "mouse_dy": self._mouse_dy,
            "event": event,
        }

    def log_key_down(self, key: str) -> None:
        self._keys[key] = 1
        self._records.append(self._snapshot(f"key_down:{key}"))

    def log_key_up(self, key: str) -> None:
        self._keys[key] = 0
        self._records.append(self._snapshot(f"key_up:{key}"))

    def log_camera_start(self, dx: int, dy: int, label: str) -> None:
        self._mouse_dx = dx
        self._mouse_dy = dy
        self._records.append(self._snapshot(f"camera_start:{label}"))

    def log_camera_end(self) -> None:
        self._mouse_dx = 0
        self._mouse_dy = 0
        self._records.append(self._snapshot("camera_end"))

    def log_event(self, event: str) -> None:
        self._records.append(self._snapshot(event))

    def to_csv(self, path: pathlib.Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            w.writeheader()
            for r in self._records:
                w.writerow(r)
        log.info("Saved %d events → %s", len(self._records), path)

    @property
    def count(self) -> int:
        return len(self._records)


# ===================================================================
# 3. Game Controller
# ===================================================================

class GameController:
    # Single keys + diagonal combos (no wa/sd — contradictory pairs)
    MOVE_OPTIONS = [
        ["w"], ["a"], ["s"], ["d"],
        ["w", "a"], ["w", "d"], ["s", "a"], ["s", "d"],
    ]

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def key_down(self, key: str) -> None:
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        pydirectinput.keyUp(key)

    def smooth_mouse_move(self, total_dx: int, total_dy: int, duration: float) -> None:
        """Smoothly move mouse over duration. No clicking."""
        steps = max(int(duration * 60), 10)
        dx_per = total_dx / steps
        dy_per = total_dy / steps
        sleep_per = duration / steps

        for _ in range(steps):
            pydirectinput.moveRel(int(dx_per), int(dy_per), relative=True)
            time.sleep(sleep_per)


# ===================================================================
# 4. Exploration Loop
# ===================================================================

class Explorer:
    """Main exploration loop with concurrent movement + camera."""

    def __init__(self, config: Config, controller: GameController,
                 event_log: EventLogger) -> None:
        self._cfg = config
        self._ctrl = controller
        self._log = event_log
        self._t0: float = 0.0
        self._paused = False
        self._stop_requested = False

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def request_stop(self) -> None:
        self._stop_requested = True

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self._log.log_event("pause" if self._paused else "resume")
        log.info("Recording %s", "PAUSED" if self._paused else "RESUMED")

    def run(self) -> None:
        self._t0 = time.perf_counter()
        self._log._t0 = self._t0
        self._log.log_event("session_start")

        while not self._stop_requested and self._elapsed() < self._cfg.duration_sec:
            if self._paused:
                time.sleep(0.1)
                continue
            self._do_cycle()

        self._log.log_event("session_end")
        log.info("Session finished. Elapsed: %.1fs", self._elapsed())

    def _pick_camera(self) -> tuple[str, int, int]:
        """Pick a camera action. Returns (label, dx_per_step, dy_per_step)."""
        candidates = self._cfg.scan_candidates
        weights = self._cfg.scan_weights
        if len(weights) != len(candidates):
            weights = [1.0 / len(candidates)] * len(candidates)

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        sens = self._cfg.mouse_sensitivity
        sens_y = self._cfg.mouse_sensitivity_y
        dur = self._cfg.scan_duration_sec
        steps = max(int(dur * 60), 10)

        if chosen == "right_90":
            total_dx = int(90 * sens)
            return "right_90", int(total_dx / steps), 0
        elif chosen == "left_90":
            total_dx = int(-90 * sens)
            return "left_90", int(total_dx / steps), 0
        elif chosen == "look_up":
            total_dy = int(-30 * sens_y)
            return "look_up", 0, int(total_dy / steps)
        elif chosen == "look_down":
            total_dy = int(30 * sens_y)
            return "look_down", 0, int(total_dy / steps)
        else:  # hold
            return "hold", 0, 0

    def _do_cycle(self) -> None:
        """One cycle: pick movement + camera, execute (possibly concurrent)."""
        cfg = self._cfg
        dur = random.uniform(cfg.min_move_duration_sec, cfg.max_move_duration_sec)

        # Pick movement keys (single or combo, e.g. ["w"] or ["w","a"])
        is_idle = random.random() < cfg.idle_probability
        move_keys = [] if is_idle else random.choice(GameController.MOVE_OPTIONS)

        # Pick camera
        do_camera = cfg.enable_camera_scan and random.random() < cfg.concurrent_probability
        cam_label, cam_dx, cam_dy = ("hold", 0, 0)
        if do_camera:
            cam_label, cam_dx, cam_dy = self._pick_camera()

        # --- Execute ---

        # Start movement (press all keys in combo)
        for k in move_keys:
            self._ctrl.key_down(k)
            self._log.log_key_down(k)

        # Start camera
        if do_camera and cam_label != "hold":
            self._log.log_camera_start(cam_dx, cam_dy, cam_label)

        # Run for duration (with mouse movement if camera active)
        steps = max(int(dur * 60), 10)
        sleep_per = dur / steps
        for _ in range(steps):
            if self._stop_requested:
                break
            if do_camera and cam_label != "hold":
                pydirectinput.moveRel(cam_dx, cam_dy, relative=True)
            time.sleep(sleep_per)

        # Stop camera
        if do_camera and cam_label != "hold":
            self._log.log_camera_end()

        # Stop movement (release all keys in combo)
        for k in move_keys:
            self._ctrl.key_up(k)
            self._log.log_key_up(k)

        # If no camera was done concurrently, do a standalone camera action
        if cfg.enable_camera_scan and not do_camera:
            cam_label, cam_dx, cam_dy = self._pick_camera()
            if cam_label != "hold":
                self._log.log_camera_start(cam_dx, cam_dy, cam_label)
                self._ctrl.smooth_mouse_move(
                    cam_dx * max(int(cfg.scan_duration_sec * 60), 10),
                    cam_dy * max(int(cfg.scan_duration_sec * 60), 10),
                    cfg.scan_duration_sec,
                )
                self._log.log_camera_end()
            else:
                self._log.log_event("camera_hold")
                time.sleep(cfg.scan_hold_duration_sec)


# ===================================================================
# 5. Data Collector
# ===================================================================

class DataCollector:
    def __init__(self, config: Config, is_first: bool = True) -> None:
        self._cfg = config
        self._is_first = is_first
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = pathlib.Path(config.output_dir) / f"session_{self._session_id}"

    def run(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        log.info("Session dir: %s", self._session_dir)

        events_csv = self._session_dir / f"session_{self._session_id}_events.csv"
        summary_path = self._session_dir / f"session_{self._session_id}_summary.json"

        controller = GameController(self._cfg)
        t0 = time.perf_counter()
        event_log = EventLogger(t0)

        explorer = Explorer(self._cfg, controller, event_log)

        keyboard.add_hotkey(self._cfg.hotkey_stop, explorer.request_stop)
        keyboard.add_hotkey(self._cfg.hotkey_pause, explorer.toggle_pause)

        if self._is_first:
            log.info("Hotkeys: %s=stop  %s=pause/resume", self._cfg.hotkey_stop, self._cfg.hotkey_pause)
            log.info("Starting in %d seconds — switch to game!", self._cfg.countdown_sec)
            for i in range(self._cfg.countdown_sec, 0, -1):
                log.info("  %d …", i)
                time.sleep(1)

        wall_clock_start = _now_iso()
        log.info("Session started: %s", wall_clock_start)

        try:
            explorer.run()
        except KeyboardInterrupt:
            log.info("Ctrl+C — saving partial data.")
            raise
        except Exception:
            log.exception("Error — saving partial data.")
        finally:
            keyboard.unhook_all()
            event_log.to_csv(events_csv)

            summary = {
                "game_name": self._cfg.game_name,
                "session_id": self._session_id,
                "wall_clock_start": wall_clock_start,
                "wall_clock_end": _now_iso(),
                "duration_sec": self._cfg.duration_sec,
                "total_events": event_log.count,
                "events_csv_path": str(events_csv),
                "config_snapshot": self._cfg.to_dict(),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            log.info("Summary → %s", summary_path)
            log.info("Done. Files in: %s", self._session_dir)


# ===================================================================
# 6. Entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cyberpunk 2077 — World Model Training Data Collector"
    )
    parser.add_argument("--config", "-c", default="config.json")
    parser.add_argument("--loops", "-n", type=int, default=0,
                        help="Number of sessions (0 = infinite)")
    args = parser.parse_args()

    if not _IS_WINDOWS:
        log.error("This script requires Windows.")
        sys.exit(1)

    cfg = Config.from_json(args.config)

    loop_num = 0
    try:
        while True:
            loop_num += 1
            if args.loops > 0 and loop_num > args.loops:
                break
            log.info("=== Session %d (duration: %ds) ===", loop_num, int(cfg.duration_sec))
            DataCollector(cfg, is_first=(loop_num == 1)).run()
            log.info("Session %d complete. Next in 3s …", loop_num)
            time.sleep(3)
    except KeyboardInterrupt:
        log.info("Stopped after %d sessions.", loop_num)


if __name__ == "__main__":
    main()
