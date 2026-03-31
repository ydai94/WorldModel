"""
Cyberpunk 2077 — World Model Training Data Collector
=====================================================
Automated first-person exploration with random movement and camera turns.
Video recording handled externally by OBS Studio.
Outputs action logs with wall-clock timestamps for OBS alignment.

Target platform: Windows 10/11, Python 3.10+
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import enum
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
    scan_after_every_move: bool = True
    scan_candidates: list[str] = field(
        default_factory=lambda: ["right_90", "left_90", "back_180", "look_up", "look_down", "hold"]
    )
    scan_weights: list[float] = field(
        default_factory=lambda: [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]
    )
    scan_duration_sec: float = 2.0
    scan_hold_duration_sec: float = 1.5
    mouse_sensitivity: float = 5.0  # pixels per degree (horizontal)
    mouse_sensitivity_y: float = 3.0  # pixels per degree (vertical)

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
# 2. Data records & logging
# ===================================================================

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


@dataclass
class ActionRecord:
    action_id: int
    action_name: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    wall_clock_start: str
    wall_clock_end: str
    params_json: str
    note: str = ""


class ActionLogger:
    _FIELDS = [
        "action_id", "action_name", "start_time_sec", "end_time_sec",
        "duration_sec", "wall_clock_start", "wall_clock_end", "params_json", "note",
    ]

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []
        self._next_id = 0

    def log(self, name: str, start: float, end: float,
            params: dict | None = None, note: str = "",
            wc_start: str = "", wc_end: str = "") -> ActionRecord:
        rec = ActionRecord(
            action_id=self._next_id,
            action_name=name,
            start_time_sec=round(start, 4),
            end_time_sec=round(end, 4),
            duration_sec=round(end - start, 4),
            wall_clock_start=wc_start or _now_iso(),
            wall_clock_end=wc_end or _now_iso(),
            params_json=json.dumps(params or {}, ensure_ascii=False),
            note=note,
        )
        self._records.append(rec)
        self._next_id += 1
        return rec

    def to_csv(self, path: pathlib.Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._FIELDS)
            w.writeheader()
            for r in self._records:
                w.writerow(dataclasses.asdict(r))
        log.info("Saved %d actions → %s", len(self._records), path)

    @property
    def count(self) -> int:
        return len(self._records)


# ===================================================================
# 3. Game Controller
# ===================================================================

class GameController:
    MOVE_KEYS = {
        "move_forward": "w",
        "move_backward": "s",
        "move_left": "a",
        "move_right": "d",
    }

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def key_down(self, key: str) -> None:
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        pydirectinput.keyUp(key)

    def smooth_mouse_move(self, total_dx: int, total_dy: int, duration: float) -> None:
        """Smoothly move mouse over duration. No clicking — just camera rotation."""
        steps = max(int(duration * 60), 10)
        dx_per = total_dx / steps
        dy_per = total_dy / steps
        sleep_per = duration / steps

        for _ in range(steps):
            pydirectinput.moveRel(int(dx_per), int(dy_per), relative=True)
            time.sleep(sleep_per)


# ===================================================================
# 4. State Machine
# ===================================================================

class State(enum.Enum):
    INIT = "init"
    RANDOM_MOVEMENT = "random_movement"
    CAMERA_OBSERVE = "camera_observe"
    END = "end"


class StateMachine:
    def __init__(self, config: Config, controller: GameController,
                 action_logger: ActionLogger) -> None:
        self._cfg = config
        self._ctrl = controller
        self._actions = action_logger
        self._state = State.INIT
        self._t0: float = 0.0
        self._paused = False
        self._stop_requested = False

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def request_stop(self) -> None:
        self._stop_requested = True

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        t = self._elapsed()
        tag = "pause" if self._paused else "resume"
        self._actions.log(tag, t, t)
        log.info("Recording %s", "PAUSED" if self._paused else "RESUMED")

    def run(self) -> None:
        self._t0 = time.perf_counter()

        while True:
            if self._stop_requested or self._elapsed() >= self._cfg.duration_sec:
                break

            if self._paused:
                time.sleep(0.1)
                continue

            if self._state == State.INIT:
                self._state = State.RANDOM_MOVEMENT
            elif self._state == State.RANDOM_MOVEMENT:
                self._do_movement()
            elif self._state == State.CAMERA_OBSERVE:
                self._do_camera_observe()

        log.info("Session finished. Elapsed: %.1fs", self._elapsed())

    def _do_movement(self) -> None:
        if not self._cfg.enable_random_movement:
            self._state = State.CAMERA_OBSERVE
            return

        if random.random() < self._cfg.idle_probability:
            dur = random.uniform(self._cfg.min_move_duration_sec, self._cfg.max_move_duration_sec)
            t = self._elapsed()
            wc = _now_iso()
            time.sleep(dur)
            self._actions.log("idle", t, self._elapsed(),
                              {"duration": round(dur, 3)}, wc_start=wc, wc_end=_now_iso())
        else:
            move_name = random.choice(list(GameController.MOVE_KEYS.keys()))
            key = GameController.MOVE_KEYS[move_name]
            dur = random.uniform(self._cfg.min_move_duration_sec, self._cfg.max_move_duration_sec)
            t = self._elapsed()
            wc = _now_iso()
            self._ctrl.key_down(key)
            time.sleep(dur)
            self._ctrl.key_up(key)
            self._actions.log(move_name, t, self._elapsed(),
                              {"key": key, "duration": round(dur, 3)},
                              wc_start=wc, wc_end=_now_iso())

        if self._cfg.enable_camera_scan and self._cfg.scan_after_every_move:
            self._state = State.CAMERA_OBSERVE
        else:
            self._state = State.RANDOM_MOVEMENT

    def _do_camera_observe(self) -> None:
        candidates = self._cfg.scan_candidates
        weights = self._cfg.scan_weights
        if len(weights) != len(candidates):
            weights = [1.0 / len(candidates)] * len(candidates)

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        t_start = self._elapsed()
        wc_start = _now_iso()
        sens = self._cfg.mouse_sensitivity
        sens_y = self._cfg.mouse_sensitivity_y
        dur = self._cfg.scan_duration_sec

        if chosen == "right_90":
            dx = int(90 * sens)
            self._ctrl.smooth_mouse_move(dx, 0, dur)
            self._actions.log("camera_turn_right_90", t_start, self._elapsed(),
                              {"direction": "right", "angle": 90, "dx": dx},
                              wc_start=wc_start, wc_end=_now_iso())

        elif chosen == "left_90":
            dx = int(-90 * sens)
            self._ctrl.smooth_mouse_move(dx, 0, dur)
            self._actions.log("camera_turn_left_90", t_start, self._elapsed(),
                              {"direction": "left", "angle": 90, "dx": dx},
                              wc_start=wc_start, wc_end=_now_iso())

        elif chosen == "back_180":
            dx = int(180 * sens)
            self._ctrl.smooth_mouse_move(dx, 0, dur)
            self._actions.log("camera_turn_back_180", t_start, self._elapsed(),
                              {"direction": "right", "angle": 180, "dx": dx},
                              wc_start=wc_start, wc_end=_now_iso())

        elif chosen == "look_up":
            dy = int(-30 * sens_y)
            self._ctrl.smooth_mouse_move(0, dy, dur)
            self._actions.log("camera_look_up", t_start, self._elapsed(),
                              {"direction": "up", "angle": 30, "dy": dy},
                              wc_start=wc_start, wc_end=_now_iso())

        elif chosen == "look_down":
            dy = int(30 * sens_y)
            self._ctrl.smooth_mouse_move(0, dy, dur)
            self._actions.log("camera_look_down", t_start, self._elapsed(),
                              {"direction": "down", "angle": 30, "dy": dy},
                              wc_start=wc_start, wc_end=_now_iso())

        elif chosen == "hold":
            time.sleep(self._cfg.scan_hold_duration_sec)
            self._actions.log("camera_hold", t_start, self._elapsed(),
                              {"duration": self._cfg.scan_hold_duration_sec},
                              wc_start=wc_start, wc_end=_now_iso())

        self._state = State.RANDOM_MOVEMENT


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

        actions_csv = self._session_dir / f"session_{self._session_id}_actions.csv"
        summary_path = self._session_dir / f"session_{self._session_id}_summary.json"

        controller = GameController(self._cfg)
        action_log = ActionLogger()

        sm = StateMachine(self._cfg, controller, action_log)

        keyboard.add_hotkey(self._cfg.hotkey_stop, sm.request_stop)
        keyboard.add_hotkey(self._cfg.hotkey_pause, sm.toggle_pause)

        if self._is_first:
            log.info("Hotkeys: %s=stop  %s=pause/resume", self._cfg.hotkey_stop, self._cfg.hotkey_pause)
            log.info("Starting in %d seconds — switch to game!", self._cfg.countdown_sec)
            for i in range(self._cfg.countdown_sec, 0, -1):
                log.info("  %d …", i)
                time.sleep(1)

        wall_clock_start = _now_iso()
        log.info("Session started: %s", wall_clock_start)

        try:
            sm.run()
        except KeyboardInterrupt:
            log.info("Ctrl+C — saving partial data.")
            raise
        except Exception:
            log.exception("Error — saving partial data.")
        finally:
            keyboard.unhook_all()
            action_log.to_csv(actions_csv)

            summary = {
                "game_name": self._cfg.game_name,
                "session_id": self._session_id,
                "wall_clock_start": wall_clock_start,
                "wall_clock_end": _now_iso(),
                "duration_sec": self._cfg.duration_sec,
                "total_actions": action_log.count,
                "actions_csv_path": str(actions_csv),
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
                        help="Number of sessions to run (0 = infinite, default: 0)")
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
            log.info("Session %d complete. Starting next in 3 seconds …", loop_num)
            time.sleep(3)
    except KeyboardInterrupt:
        log.info("Stopped after %d sessions.", loop_num)


if __name__ == "__main__":
    main()
