"""
Honkai: Star Rail — World Model Training Data Collector
========================================================
Automated gameplay control with random exploration, weighted-random camera
observation, and interaction-opportunity probing for world model training.

Video recording is handled externally by OBS Studio.
This script outputs action logs + probe logs with wall-clock timestamps
for alignment with OBS video.

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
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Platform guard — the script only works on Windows
# ---------------------------------------------------------------------------
_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import ctypes
    # Make process DPI-aware so pixel coordinates match physical screen pixels.
    # Without this, on high-DPI displays (e.g. 200% scaling), click coordinates
    # are interpreted as logical pixels and land in the wrong place.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows

    import mss
    import pydirectinput
    import keyboard

    pydirectinput.PAUSE = 0.0  # disable default inter-command pause
else:
    mss = None  # type: ignore
    pydirectinput = None  # type: ignore
    keyboard = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hsr_collector")

# ===================================================================
# 1. Configuration
# ===================================================================

@dataclass
class Config:
    """All tunable parameters.  Loaded from JSON with fallback defaults."""

    game_name: str = "HonkaiStarRail"
    duration_sec: float = 600.0
    output_dir: str = "outputs"
    monitor_index: int = 1

    # ROI for interaction-prompt detection (screen coords)
    roi_x: int = 1500
    roi_y: int = 850
    roi_w: int = 400
    roi_h: int = 250

    # Template matching
    template_paths: list[str] = field(default_factory=lambda: ["templates/interact_f_key.png"])
    template_threshold: float = 0.75

    # OCR (optional)
    enable_ocr: bool = False
    ocr_keywords: list[str] = field(
        default_factory=lambda: ["对话", "交互", "调查", "Talk", "Interact"]
    )

    # Photo mode — multi-step click sequence
    # Flow: ESC → camera icon → first person → flip camera → hide UI
    # All positions are SCREEN coordinates (fullscreen = game coordinates).
    # Use  python main.py --calibrate  to find exact positions on your setup.
    photo_esc_menu_camera_pos: list[int] = field(default_factory=lambda: [2900, 980])
    photo_first_person_pos: list[int] = field(default_factory=lambda: [2840, 1630])
    photo_flip_camera_pos: list[int] = field(default_factory=lambda: [2740, 1630])
    photo_hide_ui_pos: list[int] = field(default_factory=lambda: [2630, 1630])
    photo_step_delay: float = 0.8   # wait between each click step
    photo_mode_exit_delay: float = 1.0

    # Probe
    probe_interval_sec: float = 30.0
    probe_ui_recover_sec: float = 0.3

    # Interact key (abstracted — HSR default is F)
    interact_key: str = "f"

    # Movement
    enable_random_movement: bool = True
    min_move_duration_sec: float = 0.5
    max_move_duration_sec: float = 3.0
    min_turn_duration_sec: float = 0.3
    max_turn_duration_sec: float = 1.5
    idle_probability: float = 0.1

    # Camera observation
    enable_camera_scan: bool = True
    scan_after_every_move: bool = True
    scan_mode: str = "weighted_random"
    scan_candidates: list[str] = field(
        default_factory=lambda: ["right_90", "left_90", "back_180", "hold", "full_360"]
    )
    scan_weights: list[float] = field(
        default_factory=lambda: [0.35, 0.20, 0.25, 0.15, 0.05]
    )
    scan_duration_sec: float = 2.0
    scan_hold_duration_sec: float = 1.5
    mouse_sensitivity: float = 5.0  # pixels per degree

    # Hotkeys
    hotkey_stop: str = "F8"
    hotkey_pause: str = "F9"

    # Countdown before start
    countdown_sec: int = 5

    # Sync marker — visible camera shake at start for OBS alignment
    sync_marker_shakes: int = 3
    sync_marker_dx: int = 300
    sync_marker_interval: float = 0.3

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> Config:
        p = pathlib.Path(path)
        if not p.exists():
            log.warning("Config file %s not found — using defaults.", p)
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {fld.name for fld in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ===================================================================
# 2. Data records
# ===================================================================

@dataclass
class ActionRecord:
    action_id: int
    action_name: str
    start_time_sec: float       # relative to session start
    end_time_sec: float
    duration_sec: float
    wall_clock_start: str       # ISO 8601 for OBS alignment
    wall_clock_end: str
    params_json: str
    note: str = ""


@dataclass
class ProbeRecord:
    probe_id: int
    timestamp_sec: float        # relative to session start
    wall_clock: str             # ISO 8601
    in_photo_mode_before: bool
    template_score: float
    template_detected: bool
    ocr_enabled: bool
    ocr_text: str
    ocr_hit: bool
    interactable: bool
    note: str = ""


# ===================================================================
# 3. Loggers
# ===================================================================

def _now_iso() -> str:
    """Current wall-clock time as ISO 8601 string with milliseconds."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


class ActionLogger:
    _FIELDS = [
        "action_id", "action_name", "start_time_sec", "end_time_sec",
        "duration_sec", "wall_clock_start", "wall_clock_end", "params_json", "note",
    ]

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []
        self._next_id = 0

    def log(
        self,
        action_name: str,
        start: float,
        end: float,
        params: dict | None = None,
        note: str = "",
        wall_start: str = "",
        wall_end: str = "",
    ) -> ActionRecord:
        rec = ActionRecord(
            action_id=self._next_id,
            action_name=action_name,
            start_time_sec=round(start, 4),
            end_time_sec=round(end, 4),
            duration_sec=round(end - start, 4),
            wall_clock_start=wall_start or _now_iso(),
            wall_clock_end=wall_end or _now_iso(),
            params_json=json.dumps(params or {}, ensure_ascii=False),
            note=note,
        )
        self._records.append(rec)
        self._next_id += 1
        log.debug(
            "ACTION %04d  %-25s  %.2f–%.2f (%.2fs)",
            rec.action_id, rec.action_name, start, end, rec.duration_sec,
        )
        return rec

    def to_csv(self, path: pathlib.Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS)
            writer.writeheader()
            for r in self._records:
                writer.writerow(dataclasses.asdict(r))
        log.info("Saved %d actions → %s", len(self._records), path)

    @property
    def count(self) -> int:
        return len(self._records)


class ProbeLogger:
    _FIELDS = [
        "probe_id", "timestamp_sec", "wall_clock",
        "in_photo_mode_before", "template_score", "template_detected",
        "ocr_enabled", "ocr_text", "ocr_hit", "interactable", "note",
    ]

    def __init__(self) -> None:
        self._records: list[ProbeRecord] = []
        self._next_id = 0

    def log(self, **kwargs: Any) -> ProbeRecord:
        rec = ProbeRecord(probe_id=self._next_id, **kwargs)
        rec.timestamp_sec = round(rec.timestamp_sec, 4)
        rec.template_score = round(rec.template_score, 4)
        self._records.append(rec)
        self._next_id += 1
        tag = "HIT" if rec.interactable else "---"
        log.debug("PROBE %04d [%s] score=%.3f", rec.probe_id, tag, rec.template_score)
        return rec

    def to_csv(self, path: pathlib.Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS)
            writer.writeheader()
            for r in self._records:
                writer.writerow(dataclasses.asdict(r))
        log.info("Saved %d probes → %s", len(self._records), path)

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def hit_count(self) -> int:
        return sum(1 for r in self._records if r.interactable)


# ===================================================================
# 4. Screen Capture (probe-only, single-shot)
# ===================================================================

class ScreenCapture:
    """Single-shot screen capture via mss — used only for probe detection."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._sct: Any = None
        self._monitor: dict[str, int] = {}

    def start(self) -> None:
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if self._cfg.monitor_index >= len(monitors):
            raise RuntimeError(
                f"Monitor index {self._cfg.monitor_index} out of range "
                f"(available: 0–{len(monitors)-1})"
            )
        self._monitor = monitors[self._cfg.monitor_index]
        log.info(
            "Screen capture ready — monitor %d: %dx%d",
            self._cfg.monitor_index,
            self._monitor["width"],
            self._monitor["height"],
        )

    def grab_roi(self) -> np.ndarray:
        """Capture the ROI region of the screen and return as BGR numpy array."""
        # Grab full screen
        img = self._sct.grab(self._monitor)
        frame = np.array(img, dtype=np.uint8)[:, :, :3]  # BGRA → BGR
        # Extract ROI
        x, y = self._cfg.roi_x, self._cfg.roi_y
        w, h = self._cfg.roi_w, self._cfg.roi_h
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        x2 = min(x + w, frame.shape[1])
        y2 = min(y + h, frame.shape[0])
        if x2 <= x or y2 <= y:
            log.warning("ROI is empty after clamping (%d,%d,%d,%d)", x, y, w, h)
            return frame[0:1, 0:1]
        return frame[y:y2, x:x2]

    def stop(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None


# ===================================================================
# 5. Interaction Detector
# ===================================================================

class InteractionDetector:
    """Template matching + optional OCR on an ROI image."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._templates: list[np.ndarray] = []
        self._ocr_engine: Any = None

    def load(self) -> None:
        for tp in self._cfg.template_paths:
            p = pathlib.Path(tp)
            if not p.exists():
                log.warning("Template not found: %s — skipping.", p)
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                log.warning("Failed to read template: %s", p)
                continue
            self._templates.append(img)
            log.info("Loaded template: %s (%dx%d)", p, img.shape[1], img.shape[0])

        if not self._templates:
            log.warning("No valid templates loaded. Template matching will return 0.")

        if self._cfg.enable_ocr:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore
                self._ocr_engine = RapidOCR()
                log.info("RapidOCR engine loaded.")
            except Exception as exc:
                log.warning("OCR init failed (%s). Disabling OCR.", exc)
                self._cfg.enable_ocr = False

    def detect(self, roi_bgr: np.ndarray) -> tuple[float, bool, str, bool]:
        """Returns (template_score, template_detected, ocr_text, ocr_hit)."""
        best_score = 0.0
        if self._templates:
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            for tmpl in self._templates:
                if tmpl.shape[0] > roi_gray.shape[0] or tmpl.shape[1] > roi_gray.shape[1]:
                    continue
                result = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
        template_detected = best_score >= self._cfg.template_threshold

        ocr_text = ""
        ocr_hit = False
        if self._cfg.enable_ocr and self._ocr_engine is not None:
            try:
                result, _ = self._ocr_engine(roi_bgr)
                if result:
                    ocr_text = " ".join(line[1] for line in result)
                    ocr_hit = any(kw in ocr_text for kw in self._cfg.ocr_keywords)
            except Exception as exc:
                log.debug("OCR error: %s", exc)

        return best_score, template_detected, ocr_text, ocr_hit


# ===================================================================
# 6. Game Controller
# ===================================================================

class GameController:
    """Sends input to the game via pydirectinput (Windows SendInput)."""

    MOVE_KEYS = {
        "move_forward": "w",
        "move_backward": "s",
        "move_left": "a",
        "move_right": "d",
    }

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def tap_key(self, key: str, delay: float = 0.05) -> None:
        """Key press with deliberate hold — press() can be too fast for some games."""
        pydirectinput.keyDown(key)
        time.sleep(0.1)
        pydirectinput.keyUp(key)
        time.sleep(delay)

    def key_down(self, key: str) -> None:
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        pydirectinput.keyUp(key)

    def click_at(self, x: int, y: int) -> None:
        """Click at absolute screen coordinates with a deliberate press-hold-release."""
        pydirectinput.moveTo(x, y)
        time.sleep(0.05)
        pydirectinput.mouseDown(button="left")
        time.sleep(0.08)  # hold briefly — too-fast clicks can be ignored by game UI
        pydirectinput.mouseUp(button="left")

    def mouse_down(self) -> None:
        pydirectinput.mouseDown(button="left")

    def mouse_up(self) -> None:
        pydirectinput.mouseUp(button="left")

    def smooth_mouse_rotate(self, total_dx: int, duration: float) -> None:
        """Rotate camera by holding left-click and dragging *total_dx* pixels
        over *duration* seconds.  HSR photo mode requires click-drag."""
        steps = max(int(duration * 60), 10)
        dx_per = total_dx / steps
        sleep_per = duration / steps

        self.mouse_down()
        try:
            for _ in range(steps):
                pydirectinput.moveRel(int(dx_per), 0, relative=True)
                time.sleep(sleep_per)
        finally:
            self.mouse_up()

    def _screen_center(self) -> tuple[int, int]:
        """Approximate screen center for restoring keyboard focus."""
        # Use mss to get monitor geometry
        sct = mss.mss()
        mon = sct.monitors[self._cfg.monitor_index]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2
        sct.close()
        return cx, cy

    def enter_photo_mode(self) -> None:
        """Multi-step: ESC → camera icon → first person → flip camera → hide UI."""

        # 1. ESC to open menu
        log.info("  [enter] Step 1: ESC to open menu")
        self.tap_key("escape")
        time.sleep(1.5)

        # 2. Click camera icon in ESC menu
        pos = self._cfg.photo_esc_menu_camera_pos
        log.info("  [enter] Step 2: Click camera icon at %s", pos)
        self.click_at(pos[0], pos[1])
        time.sleep(3.0)

        # 3. Click first-person view
        pos = self._cfg.photo_first_person_pos
        log.info("  [enter] Step 3: Click first-person at %s", pos)
        self.click_at(pos[0], pos[1])
        time.sleep(1.5)

        # 4. Click flip camera (reverse camera)
        pos = self._cfg.photo_flip_camera_pos
        log.info("  [enter] Step 4: Click flip-camera at %s", pos)
        self.click_at(pos[0], pos[1])
        time.sleep(1.5)

        # 5. Click eye icon to hide UI
        pos = self._cfg.photo_hide_ui_pos
        log.info("  [enter] Step 5: Click hide-UI at %s", pos)
        self.click_at(pos[0], pos[1])
        time.sleep(1.5)

        log.info("  [enter] Photo mode ready — first-person no-UI")

    def exit_photo_mode(self) -> None:
        """Two ESC presses: 1st exits first-person/no-UI, 2nd exits photo mode."""
        log.info("  [exit] Step 1: ESC — exit first-person no-UI")
        pydirectinput.keyDown("escape")
        time.sleep(0.3)
        pydirectinput.keyUp("escape")
        time.sleep(2.5)

        log.info("  [exit] Step 2: ESC — exit photo mode to gameplay")
        pydirectinput.keyDown("escape")
        time.sleep(0.3)
        pydirectinput.keyUp("escape")
        time.sleep(2.5)

    def sync_marker(self) -> None:
        """Perform a visible camera shake pattern for OBS video alignment.
        Quick left-right-left shakes that are easy to spot in the video."""
        dx = self._cfg.sync_marker_dx
        interval = self._cfg.sync_marker_interval
        self.mouse_down()
        try:
            for _ in range(self._cfg.sync_marker_shakes):
                pydirectinput.moveRel(dx, 0, relative=True)
                time.sleep(interval)
                pydirectinput.moveRel(-dx, 0, relative=True)
                time.sleep(interval)
        finally:
            self.mouse_up()


# ===================================================================
# 7. State Machine
# ===================================================================

class State(enum.Enum):
    INIT = "init"
    RANDOM_MOVEMENT = "random_movement"
    CAMERA_OBSERVE = "camera_observe"
    DECISION_POINT = "decision_point"
    PROBE_INTERACTION = "probe_interaction"
    END = "end"


class StateMachine:
    """Core exploration + recording state machine."""

    def __init__(
        self,
        config: Config,
        controller: GameController,
        action_logger: ActionLogger,
        probe_logger: ProbeLogger,
        detector: InteractionDetector,
        screen: ScreenCapture,
    ) -> None:
        self._cfg = config
        self._ctrl = controller
        self._actions = action_logger
        self._probes = probe_logger
        self._detector = detector
        self._screen = screen

        self._state = State.INIT
        self._t0: float = 0.0
        self._last_probe: float = 0.0
        self._in_photo_mode = False
        self._paused = False
        self._stop_requested = False

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _should_probe(self) -> bool:
        return (self._elapsed() - self._last_probe) >= self._cfg.probe_interval_sec

    def set_start_time(self, t0: float) -> None:
        self._t0 = t0
        self._last_probe = 0.0

    def request_stop(self) -> None:
        self._stop_requested = True

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        tag = "pause" if self._paused else "resume"
        t = self._elapsed()
        self._actions.log(tag, t, t)
        log.info("Recording %s", "PAUSED" if self._paused else "RESUMED")

    def run(self) -> None:
        """Main blocking loop."""
        self.set_start_time(time.perf_counter())

        while True:
            if self._stop_requested or self._elapsed() >= self._cfg.duration_sec:
                self._state = State.END
                break

            if self._paused:
                time.sleep(0.1)
                continue

            if self._state == State.INIT:
                self._do_init()
            elif self._state == State.RANDOM_MOVEMENT:
                self._do_random_movement()
            elif self._state == State.CAMERA_OBSERVE:
                self._do_camera_observe()
            elif self._state == State.DECISION_POINT:
                self._do_decision()
            elif self._state == State.PROBE_INTERACTION:
                self._do_probe()

        log.info("Session finished. Elapsed: %.1fs", self._elapsed())

    # -- state handlers -------------------------------------------------

    def _do_init(self) -> None:
        # Sync marker — visible camera shake for OBS alignment
        log.info("Performing sync marker (camera shake) …")
        t = self._elapsed()
        wc_start = _now_iso()
        self._ctrl.sync_marker()
        self._actions.log("sync_marker", t, self._elapsed(),
                          {"shakes": self._cfg.sync_marker_shakes},
                          wall_start=wc_start, wall_end=_now_iso())

        log.info("Entering photo mode …")
        t = self._elapsed()
        wc_start = _now_iso()
        self._ctrl.enter_photo_mode()
        self._in_photo_mode = True
        self._actions.log("enter_photo_mode", t, self._elapsed(),
                          {"method": "click_sequence"},
                          wall_start=wc_start, wall_end=_now_iso())
        self._state = State.RANDOM_MOVEMENT

    def _do_random_movement(self) -> None:
        if not self._cfg.enable_random_movement:
            self._state = State.DECISION_POINT
            return

        if random.random() < self._cfg.idle_probability:
            dur = random.uniform(self._cfg.min_move_duration_sec, self._cfg.max_move_duration_sec)
            t_start = self._elapsed()
            wc_start = _now_iso()
            time.sleep(dur)
            self._actions.log("idle", t_start, self._elapsed(),
                              {"duration": round(dur, 3)},
                              wall_start=wc_start, wall_end=_now_iso())
        else:
            move_name = random.choice(list(GameController.MOVE_KEYS.keys()))
            key = GameController.MOVE_KEYS[move_name]
            dur = random.uniform(self._cfg.min_move_duration_sec, self._cfg.max_move_duration_sec)
            t_start = self._elapsed()
            wc_start = _now_iso()
            self._ctrl.key_down(key)
            time.sleep(dur)
            self._ctrl.key_up(key)
            self._actions.log(move_name, t_start, self._elapsed(),
                              {"key": key, "duration": round(dur, 3)},
                              wall_start=wc_start, wall_end=_now_iso())

        if self._cfg.enable_camera_scan and self._cfg.scan_after_every_move:
            self._state = State.CAMERA_OBSERVE
        else:
            self._state = State.DECISION_POINT

    def _do_camera_observe(self) -> None:
        """Pick a camera observation action via weighted random and execute."""
        candidates = self._cfg.scan_candidates
        weights = self._cfg.scan_weights
        if len(weights) != len(candidates):
            weights = [1.0 / len(candidates)] * len(candidates)

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        t_start = self._elapsed()
        wc_start = _now_iso()

        if chosen == "right_90":
            dx = int(90 * self._cfg.mouse_sensitivity)
            self._ctrl.smooth_mouse_rotate(dx, self._cfg.scan_duration_sec)
            self._actions.log("camera_turn_right_90", t_start, self._elapsed(),
                              {"direction": "right", "angle": 90, "dx": dx},
                              wall_start=wc_start, wall_end=_now_iso())

        elif chosen == "left_90":
            dx = int(-90 * self._cfg.mouse_sensitivity)
            self._ctrl.smooth_mouse_rotate(dx, self._cfg.scan_duration_sec)
            self._actions.log("camera_turn_left_90", t_start, self._elapsed(),
                              {"direction": "left", "angle": 90, "dx": dx},
                              wall_start=wc_start, wall_end=_now_iso())

        elif chosen == "back_180":
            dx = int(180 * self._cfg.mouse_sensitivity)
            self._ctrl.smooth_mouse_rotate(dx, self._cfg.scan_duration_sec)
            self._actions.log("camera_turn_back_180", t_start, self._elapsed(),
                              {"direction": "right", "angle": 180, "dx": dx},
                              wall_start=wc_start, wall_end=_now_iso())

        elif chosen == "hold":
            time.sleep(self._cfg.scan_hold_duration_sec)
            self._actions.log("camera_hold", t_start, self._elapsed(),
                              {"duration": self._cfg.scan_hold_duration_sec},
                              wall_start=wc_start, wall_end=_now_iso())

        elif chosen == "full_360":
            dx = int(360 * self._cfg.mouse_sensitivity)
            dur = self._cfg.scan_duration_sec * 2
            self._ctrl.smooth_mouse_rotate(dx, dur)
            self._actions.log("camera_scan_360", t_start, self._elapsed(),
                              {"direction": "right", "angle": 360, "dx": dx},
                              wall_start=wc_start, wall_end=_now_iso())

        else:
            log.warning("Unknown scan candidate: %s — skipping", chosen)

        self._state = State.DECISION_POINT

    def _do_decision(self) -> None:
        if self._should_probe():
            self._state = State.PROBE_INTERACTION
        else:
            self._state = State.RANDOM_MOVEMENT

    def _do_probe(self) -> None:
        """Exit photo mode → screenshot ROI → template match → re-enter photo mode."""
        was_photo = self._in_photo_mode

        # 1. Exit photo mode
        if self._in_photo_mode:
            t = self._elapsed()
            wc_start = _now_iso()
            self._ctrl.exit_photo_mode()
            self._in_photo_mode = False
            self._actions.log("exit_photo_mode", t, self._elapsed(),
                              note="probe start",
                              wall_start=wc_start, wall_end=_now_iso())

        # 2. Wait for game to fully return to normal gameplay
        log.info("  [probe] Waiting for gameplay UI to settle …")
        time.sleep(self._cfg.probe_ui_recover_sec)

        # 3. Screenshot ROI & detect
        log.info("  [probe] Taking screenshot …")
        roi = self._screen.grab_roi()
        score, tmpl_hit, ocr_text, ocr_hit = self._detector.detect(roi)
        interactable = tmpl_hit or ocr_hit

        self._probes.log(
            timestamp_sec=self._elapsed(),
            wall_clock=_now_iso(),
            in_photo_mode_before=was_photo,
            template_score=score,
            template_detected=tmpl_hit,
            ocr_enabled=self._cfg.enable_ocr,
            ocr_text=ocr_text,
            ocr_hit=ocr_hit,
            interactable=interactable,
            note="",
        )

        t_probe = self._elapsed()
        self._actions.log("probe_interaction", t_probe, t_probe,
                          {"template_score": round(score, 4), "interactable": interactable},
                          wall_start=_now_iso(), wall_end=_now_iso())

        log.info(
            "  [probe] score=%.3f  interactable=%s",
            score, interactable,
        )

        # 4. Wait before re-entering — give game time to be fully ready
        log.info("  [probe] Waiting before re-entering photo mode …")
        time.sleep(1.0)

        # 5. Re-enter photo mode
        t = self._elapsed()
        wc_start = _now_iso()
        self._ctrl.enter_photo_mode()
        self._in_photo_mode = True
        self._actions.log("enter_photo_mode", t, self._elapsed(),
                          {"method": "click_sequence"},
                          note="probe end",
                          wall_start=wc_start, wall_end=_now_iso())

        self._last_probe = self._elapsed()
        self._state = State.RANDOM_MOVEMENT


# ===================================================================
# 8. Data Collector (orchestrator)
# ===================================================================

class DataCollector:
    """Top-level orchestrator."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = pathlib.Path(config.output_dir) / f"session_{self._session_id}"

    def run(self) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        log.info("Session dir: %s", self._session_dir)

        actions_csv = self._session_dir / f"session_{self._session_id}_actions.csv"
        probes_csv = self._session_dir / f"session_{self._session_id}_probes.csv"
        summary_path = self._session_dir / f"session_{self._session_id}_summary.json"

        # Build components
        screen = ScreenCapture(self._cfg)
        controller = GameController(self._cfg)
        action_log = ActionLogger()
        probe_log = ProbeLogger()
        detector = InteractionDetector(self._cfg)
        detector.load()

        sm = StateMachine(
            config=self._cfg,
            controller=controller,
            action_logger=action_log,
            probe_logger=probe_log,
            detector=detector,
            screen=screen,
        )

        # Hotkeys
        keyboard.add_hotkey(self._cfg.hotkey_stop, sm.request_stop)
        keyboard.add_hotkey(self._cfg.hotkey_pause, sm.toggle_pause)
        log.info("Hotkeys: %s=stop  %s=pause/resume", self._cfg.hotkey_stop, self._cfg.hotkey_pause)

        # Countdown
        log.info("Starting in %d seconds — switch to game window!", self._cfg.countdown_sec)
        for i in range(self._cfg.countdown_sec, 0, -1):
            log.info("  %d …", i)
            time.sleep(1)

        # Record wall-clock start time (for OBS alignment)
        wall_clock_start = _now_iso()
        log.info("Session wall-clock start: %s", wall_clock_start)
        log.info("Make sure OBS is recording!")

        # Init screen capture (for probe screenshots)
        screen.start()

        try:
            sm.run()
        except KeyboardInterrupt:
            log.info("Interrupted by Ctrl+C — saving partial data.")
        except Exception:
            log.exception("Unexpected error — saving partial data.")
        finally:
            screen.stop()
            action_log.to_csv(actions_csv)
            probe_log.to_csv(probes_csv)

            wall_clock_end = _now_iso()

            summary = {
                "game_name": self._cfg.game_name,
                "session_id": self._session_id,
                "wall_clock_start": wall_clock_start,
                "wall_clock_end": wall_clock_end,
                "duration_sec": self._cfg.duration_sec,
                "total_actions": action_log.count,
                "total_probes": probe_log.count,
                "interactable_hits": probe_log.hit_count,
                "actions_csv_path": str(actions_csv),
                "probes_csv_path": str(probes_csv),
                "note": "Video recorded externally via OBS. Use wall_clock timestamps or sync_marker to align.",
                "config_snapshot": self._cfg.to_dict(),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            log.info("Summary → %s", summary_path)

            keyboard.unhook_all()
            log.info("Session complete.  Files in: %s", self._session_dir)


# ===================================================================
# 9. Calibration tool
# ===================================================================

def run_calibrate(config_path: str) -> None:
    """Interactive multi-round screenshot calibration tool.

    Each round: prepare game screen → press Enter → 3s countdown → screenshot.
    Screenshot saved as PNG. Open in any image viewer to read pixel coordinates.
    """
    cfg = Config.from_json(config_path)
    import mss as _mss

    sct = _mss.mss()
    monitor = sct.monitors[cfg.monitor_index]
    w, h = monitor["width"], monitor["height"]

    out_path = pathlib.Path(cfg.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  HSR Calibration Tool — Monitor {cfg.monitor_index}: {w}x{h}")
    print("=" * 60)
    print()
    print("Steps:")
    print("  1. In the game, prepare the screen you want to capture")
    print("  2. Come back here, press Enter")
    print("  3. 3s countdown → screenshot saved to outputs/")
    print("  4. Open the PNG — hover over UI elements to read coordinates")
    print()
    print("You need coordinates for:")
    print("  (a) ESC menu  → camera icon")
    print("  (b) Photo mode → eye (hide UI) button")
    print("  (c) Photo mode → camera-flip (1st person) button")
    print("  (d) Gameplay   → F interaction prompt area (for ROI)")
    print()

    round_num = 0
    while True:
        round_num += 1
        ans = input(f"[Round {round_num}] Press Enter to capture (or 'q' to quit): ").strip()
        if ans.lower() == "q":
            break

        print("  Capturing in 3 seconds — switch to the game NOW!")
        for i in range(3, 0, -1):
            print(f"    {i}...")
            time.sleep(1)

        img = sct.grab(monitor)
        frame = np.array(img, dtype=np.uint8)[:, :, :3]

        filename = f"calibrate_{round_num}.png"
        save_path = out_path / filename
        cv2.imwrite(str(save_path), frame)
        print(f"  Saved → {save_path}  ({w}x{h})")
        print()

    sct.close()

    print()
    print("Update config.json with your coordinates:")
    print('  "photo_esc_menu_camera_pos": [x, y],')
    print('  "photo_hide_ui_pos": [x, y],')
    print('  "photo_first_person_pos": [x, y],')
    print('  "roi_x": x, "roi_y": y, "roi_w": w, "roi_h": h')
    print()
    print("Tip: On Windows, open the PNG in Paint — coordinates show")
    print("     in the bottom-left status bar as you move the mouse.")


# ===================================================================
# 10. Entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Honkai: Star Rail — World Model Training Data Collector"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to config JSON (default: config.json)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Capture screenshots to find UI element coordinates",
    )
    args = parser.parse_args()

    if not _IS_WINDOWS:
        log.error("This script requires Windows (pydirectinput / mss / keyboard).")
        sys.exit(1)

    if args.calibrate:
        run_calibrate(args.config)
        return

    cfg = Config.from_json(args.config)
    log.info("Game: %s | Duration: %ds", cfg.game_name, int(cfg.duration_sec))

    DataCollector(cfg).run()


if __name__ == "__main__":
    main()
