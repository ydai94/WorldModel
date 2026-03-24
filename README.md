# Honkai: Star Rail — World Model Training Data Collector

Automated gameplay recording tool for collecting world model training data from Honkai: Star Rail on Windows. The script records continuous video, executes random exploration movements with diverse camera observations, and periodically probes for interaction opportunities — all while maintaining precise action and probe logs aligned to the video timeline.

## Requirements

- Windows 10 / 11
- Python 3.10+
- Honkai: Star Rail (PC version)
- Run as **Administrator** (required for `pydirectinput` to send input to the game)

## Installation

```bash
pip install -r requirements.txt
```

For optional OCR support:
```bash
pip install rapidocr-onnxruntime
```

## Quick Start

1. Prepare a template image (see below)
2. Configure `config.json` (see below)
3. Launch Honkai: Star Rail, enter an explorable area
4. Run the script **as Administrator**:

```bash
python main.py
```

5. Switch to the game window during the 5-second countdown
6. The script will automatically:
   - Enter photo mode
   - Move randomly while recording
   - Perform camera observations (turns, holds, scans) after each movement
   - Exit photo mode periodically to probe for interaction prompts
   - Stop after 10 minutes (or press **F8** to stop early)
7. Find your data in `outputs/session_YYYYMMDD_HHMMSS/`

## Preparing the Interaction Template Image

The script detects interaction opportunities by matching a template image against a region of the screen.

1. In the game, walk up to an NPC or interactable object so the prompt appears (e.g., "对话" / "Talk" / "F" icon)
2. Take a screenshot (Win+Shift+S or PrtSc)
3. Crop **just the interaction prompt icon/text** — keep it small and distinct
4. Save as `templates/interact_template.png`
5. You can add multiple templates by adding paths to `template_paths` in config

Tips:
- Use grayscale-friendly icons (the matching converts to grayscale)
- Crop tightly — less background = better matching
- A typical size: 40×40 to 100×50 pixels

## Determining the ROI

The ROI (Region of Interest) is the screen area where the interaction prompt appears. This avoids false positives from other UI elements.

1. Take a screenshot while the interaction prompt is visible
2. Note the pixel coordinates of the prompt area
3. Set in `config.json`:
   - `roi_x`, `roi_y`: top-left corner of the ROI
   - `roi_w`, `roi_h`: width and height

For 1920×1080 resolution, the interaction prompt in HSR typically appears near the center-right of the screen. A good starting point:
```json
"roi_x": 880, "roi_y": 480, "roi_w": 200, "roi_h": 120
```

## Configuring Photo Mode Toggle Key

Honkai: Star Rail's photo mode / camera mode key may vary. Check your in-game key bindings.

- Default in config: `"photo_mode_toggle_key": "["` (left bracket)
- If your game uses a different key, update this in `config.json`
- The same key is used to both enter and exit photo mode

## Configuration Reference

See `config.json` for all fields. Key parameters:

| Field | Description | Default |
|-------|-------------|---------|
| `duration_sec` | Recording duration in seconds | 600 |
| `fps` | Frames per second for video capture | 15 |
| `photo_mode_toggle_key` | Key to toggle photo/camera mode | `[` |
| `probe_interval_sec` | Seconds between interaction probes | 30 |
| `scan_candidates` | Camera observation types | `["right_90","left_90","back_180","hold","full_360"]` |
| `scan_weights` | Probability weights for each observation type | `[0.35, 0.20, 0.25, 0.15, 0.05]` |
| `scan_duration_sec` | Duration of each camera turn | 2.0 |
| `mouse_sensitivity` | Pixels per degree for mouse rotation | 5.0 |
| `template_threshold` | Minimum match score to consider detected | 0.75 |
| `idle_probability` | Chance of idle instead of movement | 0.1 |
| `hotkey_stop` | Key to stop recording | F8 |
| `hotkey_pause` | Key to pause/resume | F9 |

## Output Files

Each session creates a directory `outputs/session_YYYYMMDD_HHMMSS/` containing:

### 1. Video (`session_*.mp4`)
Continuous screen recording at the configured FPS. Contains all gameplay including movements, camera observations, and brief UI appearances during probes.

### 2. Action Log (`session_*_actions.csv`)
Every action with precise timestamps relative to the recording start:

| Column | Description |
|--------|-------------|
| `action_id` | Sequential ID |
| `action_name` | e.g., `move_forward`, `camera_turn_right_90`, `enter_photo_mode` |
| `start_time_sec` | Start time relative to recording start |
| `end_time_sec` | End time relative to recording start |
| `duration_sec` | Duration in seconds |
| `params_json` | JSON parameters (key pressed, angle, etc.) |
| `note` | Optional annotation |

Action types: `move_forward`, `move_backward`, `move_left`, `move_right`, `idle`, `camera_turn_right_90`, `camera_turn_left_90`, `camera_turn_back_180`, `camera_hold`, `camera_scan_360`, `enter_photo_mode`, `exit_photo_mode`, `probe_interaction`, `pause`, `resume`

### 3. Probe Log (`session_*_probes.csv`)
Results of each interaction-opportunity probe:

| Column | Description |
|--------|-------------|
| `probe_id` | Sequential ID |
| `timestamp_sec` | Probe time relative to recording start |
| `frame_index` | Video frame number at probe time |
| `in_photo_mode_before` | Whether photo mode was active before probe |
| `template_score` | Best template match score (0–1) |
| `template_detected` | Whether score exceeded threshold |
| `ocr_enabled` | Whether OCR was enabled |
| `ocr_text` | OCR-detected text (if enabled) |
| `ocr_hit` | Whether OCR found a keyword |
| `interactable` | Final judgment: template_detected OR ocr_hit |
| `note` | Optional annotation |

### 4. Summary (`session_*_summary.json`)
Session metadata including total frames, actions, probes, hits, file paths, and a snapshot of the config used.

## Aligning Actions/Probes with Video

All timestamps in `actions.csv` and `probes.csv` are **relative to the recording start** (in seconds). To find the corresponding video frame:

```
frame_number = timestamp_sec × fps
```

For example, if `start_time_sec = 12.5` and `fps = 15`:
```
frame = 12.5 × 15 = 187 (approximately frame 187–188)
```

The `probes.csv` also includes `frame_index` for direct frame-level alignment.

### Usage in training pipelines

```python
import csv, cv2

cap = cv2.VideoCapture("outputs/session_.../session_..._actions.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

with open("outputs/session_.../session_..._actions.csv") as f:
    actions = list(csv.DictReader(f))

for action in actions:
    start_frame = int(float(action["start_time_sec"]) * fps)
    end_frame = int(float(action["end_time_sec"]) * fps)
    # Extract frames [start_frame, end_frame] and pair with action label
```

## Hotkeys

| Key | Function |
|-----|----------|
| **F8** | Stop recording and save all data |
| **F9** | Pause / resume recording |

## Troubleshooting

- **Keys not working in game**: Run the script as Administrator
- **Black frames**: Make sure the game is not minimized; use borderless windowed mode
- **Low FPS in video**: Reduce `fps` in config or ensure your system can handle the capture rate
- **Template not matching**: Lower `template_threshold` (e.g., to 0.6) or re-crop the template
- **Mouse sensitivity wrong**: Adjust `mouse_sensitivity` — higher values = larger camera turns per degree. Test with a short recording first
