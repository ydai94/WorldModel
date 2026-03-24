"""
Honkai: Star Rail — World Model Training Data Collector
========================================================
Automated gameplay recording with random exploration, weighted-random camera
observation, and interaction-opportunity probing for world model training.

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

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Platform guard — the script only works on Windows
# ---------------------------------------------------------------------------
_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    import mss
    import pydirectinput
    import keyboard

    pydirectinput.PAUSE = 0.0  # disable default inter-command pause
else:
    # Allow importing on non-Windows for linting / testing data classes
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
    fps: int = 15
    output_dir: str = "outputs"
    monitor_index: int = 1

    # ROI for interaction-prompt detection (screen coords)
    roi_x: int = 900
    roi_y: int = 500
    roi_w: int = 200
    roi_h: int = 100

    # Template matching
    template_paths: list[str] = field(default_factory=lambda: ["templates/interact_template.png"])
    template_threshold: float = 0.75

    # OCR (optional)
    enable_ocr: bool = False
    ocr_keywords: list[str] = field(
        default_factory=lambda: ["对话", "交互", "调查", "Talk", "Interact"]
    )

    # Photo mode — multi-step click sequence
    # Flow: ESC → click camera icon → click eye (hide UI) → click flip (1st person)
    # All positions are SCREEN coordinates (fullscreen = game coordinates).
    # Use  python main.py --calibrate  to find exact positions on your setup.
    photo_esc_menu_camera_pos: list[int] = field(default_factory=lambda: [2900, 980])
    photo_hide_ui_pos: list[int] = field(default_factory=lambda: [2630, 1630])
    photo_first_person_pos: list[int] = field(default_factory=lambda: [2840, 1630])
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
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    params_json: str
    note: str = ""


@dataclass
class ProbeRecord:
    probe_id: int
    timestamp_sec: float
    frame_index: int
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

class ActionLogger:
    _FIELDS = [
        "action_id", "action_name", "start_time_sec", "end_time_sec",
        "duration_sec", "params_json", "note",
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
    ) -> ActionRecord:
        rec = ActionRecord(
            action_id=self._next_id,
            action_name=action_name,
            start_time_sec=round(start, 4),
            end_time_sec=round(end, 4),
            duration_sec=round(end - start, 4),
            params_json=json.dumps(params or {}, ensure_ascii=False),
            note=note,
        )
        self._records.append(rec)
        self._next_id += 1
        log.debug("ACTION %04d  %-25s  %.2f–%.2f (%.2fs)", rec.action_id, rec.action_name, start, end, rec.duration_sec)
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
        "probe_id", "timestamp_sec", "frame_index",
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
# 4. Screen Recorder
# ===================================================================

class ScreenRecorder:
    """Captures screen frames via *mss* and writes to video via OpenCV."""

    def __init__(self, config: Config, output_path: pathlib.Path) -> None:
        self._cfg = config
        self._output_path = output_path
        self._sct: Any = None
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._monitor: dict[str, int] = {}
        self._width = 0
        self._height = 0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if self._cfg.monitor_index >= len(monitors):
            raise RuntimeError(
                f"Monitor index {self._cfg.monitor_index} out of range "
                f"(available: 0–{len(monitors)-1})"
            )
        self._monitor = monitors[self._cfg.monitor_index]
        self._width = self._monitor["width"]
        self._height = self._monitor["height"]
        log.info("Capture monitor %d: %dx%d", self._cfg.monitor_index, self._width, self._height)

        # Try codecs in order
        codec_ext_pairs = [
            ("mp4v", ".mp4"),
            ("MJPG", ".avi"),
            ("XVID", ".avi"),
        ]
        for codec, ext in codec_ext_pairs:
            out = self._output_path.with_suffix(ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(out), fourcc, self._cfg.fps, (self._width, self._height)
            )
            if writer.isOpened():
                self._writer = writer
                self._output_path = out
                log.info("VideoWriter: codec=%s  path=%s", codec, out)
                return
            writer.release()

        raise RuntimeError("Failed to initialise VideoWriter with any codec.")

    # ------------------------------------------------------------------
    def capture_frame(self) -> np.ndarray:
        img = self._sct.grab(self._monitor)
        frame = np.array(img, dtype=np.uint8)[:, :, :3]  # BGRA → BGR (drop A)
        # mss returns BGRA where B,G,R order already matches OpenCV BGR
        self._frame_count += 1
        return frame

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer:
            self._writer.write(frame)

    def capture_and_write(self) -> np.ndarray:
        frame = self.capture_frame()
        self.write_frame(frame)
        return frame

    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        x, y = self._cfg.roi_x, self._cfg.roi_y
        w, h = self._cfg.roi_w, self._cfg.roi_h
        # Clamp
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        x2 = min(x + w, frame.shape[1])
        y2 = min(y + h, frame.shape[0])
        if x2 <= x or y2 <= y:
            log.warning("ROI is empty after clamping (%d,%d,%d,%d)", x, y, w, h)
            return frame[0:1, 0:1]
        return frame[y:y2, x:x2]

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def output_path(self) -> pathlib.Path:
        return self._output_path

    def stop(self) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None
            log.info("VideoWriter released.  Total frames: %d", self._frame_count)
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
        # Load templates
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

        # OCR
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
        # Template matching
        best_score = 0.0
        if self._templates:
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            for tmpl in self._templates:
                # Skip if template is larger than ROI
                if tmpl.shape[0] > roi_gray.shape[0] or tmpl.shape[1] > roi_gray.shape[1]:
                    continue
                result = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
        template_detected = best_score >= self._cfg.template_threshold

        # OCR
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

    # Map abstract movement names → keys
    MOVE_KEYS = {
        "move_forward": "w",
        "move_backward": "s",
        "move_left": "a",
        "move_right": "d",
    }

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def tap_key(self, key: str, delay: float = 0.05) -> None:
        pydirectinput.press(key)
        time.sleep(delay)

    def key_down(self, key: str) -> None:
        pydirectinput.keyDown(key)

    def key_up(self, key: str) -> None:
        pydirectinput.keyUp(key)

    def mouse_move_relative(self, dx: int, dy: int = 0) -> None:
        pydirectinput.moveRel(dx, dy, relative=True)

    def click_at(self, x: int, y: int) -> None:
        """Click at absolute screen coordinates."""
        pydirectinput.click(x, y)

    def mouse_down(self) -> None:
        """Press and hold left mouse button (for drag operations)."""
        pydirectinput.mouseDown(button="left")

    def mouse_up(self) -> None:
        """Release left mouse button."""
        pydirectinput.mouseUp(button="left")

    # ------------------------------------------------------------------
    # Smooth camera rotation — holds left mouse button while dragging,
    # because HSR photo mode requires click-drag for camera rotation.
    # ------------------------------------------------------------------
    def smooth_mouse_rotate(
        self,
        total_dx: int,
        duration: float,
        capture_fn: Any = None,
    ) -> None:
        """Rotate camera by *total_dx* pixels over *duration* seconds.

        Holds left mouse button during the drag (required by HSR photo mode).
        If *capture_fn* is provided it is called every ~frame_interval to keep
        recording video during the rotation.
        """
        steps = max(int(duration * 60), 10)
        dx_per = total_dx / steps
        sleep_per = duration / steps
        frame_interval = 1.0 / self._cfg.fps if self._cfg.fps > 0 else 0.066
        last_capture = time.perf_counter()

        self.mouse_down()
        try:
            for _ in range(steps):
                pydirectinput.moveRel(int(dx_per), 0, relative=True)
                now = time.perf_counter()
                if capture_fn and (now - last_capture) >= frame_interval:
                    capture_fn()
                    last_capture = time.perf_counter()
                remaining = sleep_per - (time.perf_counter() - now)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self.mouse_up()

    def enter_photo_mode(self) -> None:
        """Multi-step: ESC → click camera → click eye (hide UI) → click flip (1st person)."""
        delay = self._cfg.photo_step_delay

        # 1. ESC to open menu
        self.tap_key("escape")
        time.sleep(delay)

        # 2. Click camera icon in ESC menu right sidebar
        pos = self._cfg.photo_esc_menu_camera_pos
        self.click_at(pos[0], pos[1])
        time.sleep(delay)

        # 3. Click eye icon to hide UI
        pos = self._cfg.photo_hide_ui_pos
        self.click_at(pos[0], pos[1])
        time.sleep(0.3)

        # 4. Click camera-flip icon for first-person view
        pos = self._cfg.photo_first_person_pos
        self.click_at(pos[0], pos[1])
        time.sleep(0.3)

    def exit_photo_mode(self) -> None:
        """Two ESC presses: 1st exits first-person/no-UI → photo UI, 2nd exits photo mode → gameplay."""
        # 1st ESC: exit first-person no-UI back to photo mode with UI
        self.tap_key("escape")
        time.sleep(self._cfg.photo_step_delay)
        # 2nd ESC: exit photo mode back to gameplay
        self.tap_key("escape")
        time.sleep(self._cfg.photo_mode_exit_delay)


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
        recorder: ScreenRecorder,
    ) -> None:
        self._cfg = config
        self._ctrl = controller
        self._actions = action_logger
        self._probes = probe_logger
        self._detector = detector
        self._rec = recorder

        self._state = State.INIT
        self._t0: float = 0.0  # recording start (perf_counter)
        self._last_probe: float = 0.0
        self._in_photo_mode = False
        self._paused = False
        self._stop_requested = False

    # -- helpers --------------------------------------------------------
    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def _capture(self) -> np.ndarray:
        return self._rec.capture_and_write()

    def _should_probe(self) -> bool:
        return (self._elapsed() - self._last_probe) >= self._cfg.probe_interval_sec

    # -- public ---------------------------------------------------------
    def set_start_time(self, t0: float) -> None:
        self._t0 = t0
        self._last_probe = 0.0

    def request_stop(self) -> None:
        self._stop_requested = True

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        tag = "pause" if self._paused else "resume"
        self._actions.log(tag, self._elapsed(), self._elapsed())
        log.info("Recording %s", "PAUSED" if self._paused else "RESUMED")

    def run(self) -> None:
        """Main blocking loop.  Runs until duration expires or stop requested."""
        self.set_start_time(time.perf_counter())
        frame_interval = 1.0 / self._cfg.fps

        while True:
            tick_start = time.perf_counter()

            # Termination check
            if self._stop_requested or self._elapsed() >= self._cfg.duration_sec:
                self._state = State.END
                break

            if self._paused:
                # Still capture frames so the video has no gap
                self._capture()
                elapsed_tick = time.perf_counter() - tick_start
                if elapsed_tick < frame_interval:
                    time.sleep(frame_interval - elapsed_tick)
                continue

            # Execute current state
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

        # End
        log.info("Recording finished. Elapsed: %.1fs", self._elapsed())

    # -- state handlers -------------------------------------------------

    def _do_init(self) -> None:
        log.info("Entering photo mode …")
        t = self._elapsed()
        self._ctrl.enter_photo_mode()
        self._in_photo_mode = True
        self._actions.log("enter_photo_mode", t, self._elapsed(),
                          {"method": "click_sequence"})
        self._state = State.RANDOM_MOVEMENT

    def _do_random_movement(self) -> None:
        if not self._cfg.enable_random_movement:
            self._state = State.DECISION_POINT
            return

        # Pick action
        if random.random() < self._cfg.idle_probability:
            # Idle
            dur = random.uniform(self._cfg.min_move_duration_sec, self._cfg.max_move_duration_sec)
            t_start = self._elapsed()
            self._hold_with_capture(duration=dur)
            self._actions.log("idle", t_start, self._elapsed(), {"duration": dur})
        else:
            move_name = random.choice(list(GameController.MOVE_KEYS.keys()))
            key = GameController.MOVE_KEYS[move_name]
            dur = random.uniform(self._cfg.min_move_duration_sec, self._cfg.max_move_duration_sec)
            t_start = self._elapsed()
            self._hold_key_with_capture(key, dur)
            self._actions.log(move_name, t_start, self._elapsed(), {"key": key, "duration": round(dur, 3)})

        # Next state
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

        if chosen == "right_90":
            dx = int(90 * self._cfg.mouse_sensitivity)
            self._ctrl.smooth_mouse_rotate(dx, self._cfg.scan_duration_sec, capture_fn=self._capture)
            self._actions.log("camera_turn_right_90", t_start, self._elapsed(),
                              {"direction": "right", "angle": 90, "dx": dx})

        elif chosen == "left_90":
            dx = int(-90 * self._cfg.mouse_sensitivity)
            self._ctrl.smooth_mouse_rotate(dx, self._cfg.scan_duration_sec, capture_fn=self._capture)
            self._actions.log("camera_turn_left_90", t_start, self._elapsed(),
                              {"direction": "left", "angle": 90, "dx": dx})

        elif chosen == "back_180":
            dx = int(180 * self._cfg.mouse_sensitivity)
            self._ctrl.smooth_mouse_rotate(dx, self._cfg.scan_duration_sec, capture_fn=self._capture)
            self._actions.log("camera_turn_back_180", t_start, self._elapsed(),
                              {"direction": "right", "angle": 180, "dx": dx})

        elif chosen == "hold":
            self._hold_with_capture(self._cfg.scan_hold_duration_sec)
            self._actions.log("camera_hold", t_start, self._elapsed(),
                              {"duration": self._cfg.scan_hold_duration_sec})

        elif chosen == "full_360":
            dx = int(360 * self._cfg.mouse_sensitivity)
            dur = self._cfg.scan_duration_sec * 2  # slower for full rotation
            self._ctrl.smooth_mouse_rotate(dx, dur, capture_fn=self._capture)
            self._actions.log("camera_scan_360", t_start, self._elapsed(),
                              {"direction": "right", "angle": 360, "dx": dx})

        else:
            log.warning("Unknown scan candidate: %s — skipping", chosen)

        self._state = State.DECISION_POINT

    def _do_decision(self) -> None:
        if self._should_probe():
            self._state = State.PROBE_INTERACTION
        else:
            self._state = State.RANDOM_MOVEMENT

    def _do_probe(self) -> None:
        """Exit photo mode → detect interaction → re-enter photo mode."""
        was_photo = self._in_photo_mode

        # 1. Exit photo mode
        if self._in_photo_mode:
            t = self._elapsed()
            self._ctrl.exit_photo_mode()
            self._in_photo_mode = False
            self._actions.log("exit_photo_mode", t, self._elapsed(),
                              note="probe start")

        # 2. Wait for UI to recover (capture frames meanwhile)
        self._hold_with_capture(self._cfg.probe_ui_recover_sec)

        # 3. Capture & detect
        frame = self._capture()
        roi = self._rec.get_roi(frame)
        score, tmpl_hit, ocr_text, ocr_hit = self._detector.detect(roi)

        interactable = tmpl_hit or ocr_hit

        self._probes.log(
            timestamp_sec=self._elapsed(),
            frame_index=self._rec.frame_count,
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
                          {"template_score": round(score, 4), "interactable": interactable})

        log.info(
            "Probe @ %.1fs — score=%.3f  interactable=%s",
            self._elapsed(), score, interactable,
        )

        # 4. Re-enter photo mode
        t = self._elapsed()
        self._ctrl.enter_photo_mode()
        self._in_photo_mode = True
        self._actions.log("enter_photo_mode", t, self._elapsed(),
                          {"method": "click_sequence"}, note="probe end")

        self._last_probe = self._elapsed()
        self._state = State.RANDOM_MOVEMENT

    # -- utilities ------------------------------------------------------

    def _hold_key_with_capture(self, key: str, duration: float) -> None:
        """Hold *key* for *duration* seconds while continuing to capture frames."""
        frame_interval = 1.0 / self._cfg.fps
        self._ctrl.key_down(key)
        end = time.perf_counter() + duration
        last_cap = 0.0
        try:
            while time.perf_counter() < end:
                if self._stop_requested:
                    break
                now = time.perf_counter()
                if (now - last_cap) >= frame_interval:
                    self._capture()
                    last_cap = time.perf_counter()
                remaining = min(frame_interval * 0.5, end - time.perf_counter())
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._ctrl.key_up(key)

    def _hold_with_capture(self, duration: float) -> None:
        """Wait for *duration* seconds while capturing frames."""
        frame_interval = 1.0 / self._cfg.fps
        end = time.perf_counter() + duration
        last_cap = 0.0
        while time.perf_counter() < end:
            if self._stop_requested:
                break
            now = time.perf_counter()
            if (now - last_cap) >= frame_interval:
                self._capture()
                last_cap = time.perf_counter()
            remaining = min(frame_interval * 0.5, end - time.perf_counter())
            if remaining > 0:
                time.sleep(remaining)


# ===================================================================
# 8. Data Collector (orchestrator)
# ===================================================================

class DataCollector:
    """Top-level orchestrator: sets up components, runs the state machine,
    and saves all output artefacts."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = pathlib.Path(config.output_dir) / f"session_{self._session_id}"

    def run(self) -> None:
        # 1. Prepare output dir
        self._session_dir.mkdir(parents=True, exist_ok=True)
        log.info("Session dir: %s", self._session_dir)

        video_path = self._session_dir / f"session_{self._session_id}.mp4"
        actions_csv = self._session_dir / f"session_{self._session_id}_actions.csv"
        probes_csv = self._session_dir / f"session_{self._session_id}_probes.csv"
        summary_path = self._session_dir / f"session_{self._session_id}_summary.json"

        # 2. Build components
        recorder = ScreenRecorder(self._cfg, video_path)
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
            recorder=recorder,
        )

        # 3. Hotkeys
        keyboard.add_hotkey(self._cfg.hotkey_stop, sm.request_stop)
        keyboard.add_hotkey(self._cfg.hotkey_pause, sm.toggle_pause)
        log.info("Hotkeys: %s=stop  %s=pause/resume", self._cfg.hotkey_stop, self._cfg.hotkey_pause)

        # 4. Countdown
        log.info("Starting in %d seconds — switch to game window!", self._cfg.countdown_sec)
        for i in range(self._cfg.countdown_sec, 0, -1):
            log.info("  %d …", i)
            time.sleep(1)

        # 5. Start recording
        recorder.start()
        actual_video_path = recorder.output_path  # may differ if codec fallback

        try:
            sm.run()
        except KeyboardInterrupt:
            log.info("Interrupted by Ctrl+C — saving partial data.")
        except Exception:
            log.exception("Unexpected error — saving partial data.")
        finally:
            recorder.stop()
            action_log.to_csv(actions_csv)
            probe_log.to_csv(probes_csv)

            # Summary
            summary = {
                "game_name": self._cfg.game_name,
                "session_id": self._session_id,
                "duration_sec": self._cfg.duration_sec,
                "fps": self._cfg.fps,
                "total_frames": recorder.frame_count,
                "total_actions": action_log.count,
                "total_probes": probe_log.count,
                "interactable_hits": probe_log.hit_count,
                "output_video_path": str(actual_video_path),
                "actions_csv_path": str(actions_csv),
                "probes_csv_path": str(probes_csv),
                "config_snapshot": self._cfg.to_dict(),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            log.info("Summary → %s", summary_path)

            keyboard.unhook_all()
            log.info("Session complete.  Files in: %s", self._session_dir)


# ===================================================================
# 9. Entry point
# ===================================================================

def run_calibrate(config_path: str) -> None:
    """Interactive multi-round screenshot calibration tool.

    Flow:
      1. Prompts you in the terminal (not in the game)
      2. You switch to the game and prepare the screen you want to capture
      3. Press Enter in the terminal → 3-second countdown → screenshot
      4. Screenshot saved as PNG → open it in any image viewer (e.g. Paint)
         to read pixel coordinates
      5. Repeat for as many screens as you need (ESC menu, photo mode, etc.)

    No OpenCV window needed — just use your image viewer to get coordinates.
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
    print("How it works:")
    print("  1. Switch to the game, prepare the screen you want")
    print("  2. Come back here and press Enter")
    print("  3. 3-second countdown → screenshot captured")
    print("  4. Open the saved PNG in Paint / image viewer")
    print("     → hover over UI elements to read their coordinates")
    print("  5. Put those coordinates into config.json")
    print()
    print("You need coordinates for:")
    print("  (a) ESC menu → camera icon position")
    print("  (b) Photo mode → eye (hide UI) button position")
    print("  (c) Photo mode → camera-flip (1st person) button position")
    print("  (d) Gameplay → F interaction prompt area (for ROI)")
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
        print(f"  Saved → {save_path}")
        print(f"  Open this file in Paint (or any viewer) and note the pixel coordinates.")
        print()

    sct.close()

    print()
    print("Done. Update config.json with the coordinates you found:")
    print('  "photo_esc_menu_camera_pos": [x, y],')
    print('  "photo_hide_ui_pos": [x, y],')
    print('  "photo_first_person_pos": [x, y],')
    print('  "roi_x": ..., "roi_y": ..., "roi_w": ..., "roi_h": ...')
    print()


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
        help="Capture a screenshot and click to find UI element coordinates",
    )
    args = parser.parse_args()

    if not _IS_WINDOWS:
        log.error("This script requires Windows (pydirectinput / mss / keyboard).")
        sys.exit(1)

    if args.calibrate:
        run_calibrate(args.config)
        return

    cfg = Config.from_json(args.config)
    log.info("Game: %s | Duration: %ds | FPS: %d", cfg.game_name, int(cfg.duration_sec), cfg.fps)

    DataCollector(cfg).run()


if __name__ == "__main__":
    main()
