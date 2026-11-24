# Volleyball Vision Analytics: Referee Pose & Gesture Detection

This module implements **Goal 1 (Approach 1)** of the **Volleyball Vision Analytics** project: automatically locating the end of rallies in beach volleyball broadcasts (*Beach Volleyball World*) by detecting the referee's point-awarding arm gestures.

---

## 1. Pipeline Architecture (How It Works)

Instead of running heavy video models across the entire stadium, the pipeline narrows down the problem in five clear steps:

1. **Static Region of Interest (ROI) Selection:**
   Main broadcast cameras in beach volleyball remain mostly fixed during play. For each match (or camera angle), we define a bounding box `(x1, y1, x2, y2)` around the referee's stand once. Cropping to this ROI prevents players on the court and spectators in the background from distracting the pose model.

2. **Lighting-Invariant Camera Cut Filter (HSV Color Palette):**
   Sports broadcasts frequently cut to replays, close-ups, timeouts, or the crowd. To skip irrelevant frames, each frame is compared against a reference frame of the main court view. Because beach volleyball is played outdoors—where sunlight, clouds, or rain can suddenly change brightness and warmth—we compare **Hue and Saturation color palettes** (in HSV space, ignoring brightness/Value). A sharp drop in color-palette similarity indicates a camera cut and sets `referee_visible = 0`.

3. **Pose Estimation (`YOLO-pose`):**
   On valid main-camera frames, a pretrained **YOLO-pose** model runs inside the referee ROI to detect upper-body keypoints: shoulders, elbows, and wrists.

4. **Feature Engineering over Short Windows (2–5 Seconds):**
   A point-awarding signal is not a single static pose—it is a motion where the referee raises or extends an arm and holds it. From the raw keypoints, we compute geometric features (`left_arm_angle`, `right_arm_angle`) and aggregate temporal dynamics (`gesture_change`, peak extension, and hold duration) across **2–5 second sliding windows**.

5. **ML Gesture Classification:**
   A lightweight tabular machine learning classifier (e.g., Random Forest or Logistic Regression) trained on manually annotated referee signals predicts whether a point-awarding gesture occurred (`referee_signal = 1`) or not (`0`).

---

## 2. Module Outputs

The pipeline produces two artifacts:

1. **Structured Dataset (`referee_df.csv`):**
   Used for model training, quantitative evaluation against true rally-end timestamps, and comparison with the video-classification approach:
   * `match_id`: Unique broadcast identifier.
   * `observation_time`: Timestamp (in seconds) of the observation/window.
   * `referee_visible`: Binary flag (`1` if main camera is active and referee keypoints are detected, `0` otherwise).
   * `left_arm_angle` / `right_arm_angle`: Estimated arm angles in degrees ($0^\circ\text{--}180^\circ$).
   * `gesture_change`: Temporal change indicator comparing current arm angles against preceding frames.
   * `referee_signal`: Binary target/prediction (`1` = point-awarding gesture detected, `0` = neutral stance).

2. **Annotated Visualization Video (`demo_overlay.mp4`):**
   A rendered video clip showing the ROI bounding box, `YOLO-pose` skeleton keypoints on the referee, live arm angle readouts, and an on-screen alert banner when `referee_signal = 1` triggers. Used for visual debugging and presentation at the **Poster Session**.

---

## 3. Training & Evaluation Criteria

* **Strict Match-Level Split (Zero Data Leakage):** Entire matches are assigned exclusively to either the training set or the test set. Clips from the same match are never mixed across splits.
* **Two-Stage Development:** Code and feature extraction are first validated locally on short 5-minute test clips before scaling up to full matches on cloud GPU compute.
* **Ground Truth & Metrics:** Because a referee signals *after* a rally ends, predictions are evaluated against manually annotated **actual rally-end timestamps** on unseen test matches:
  * **False Positive Rate (FPR):** Target $< 15\%$ (avoiding false rally-end triggers during ongoing play).
  * **Recall:** Share of actual rally endings successfully detected.
  * **Timing Lag:** Average delay (in seconds) between the physical rally end and the detected referee gesture.

---

## 4. Handling Uncertain Situations (Edge Cases)

| Uncertain Situation | Pipeline Behavior | Data Representation |
| :--- | :--- | :--- |
| **Outdoor weather/lighting shifts (sun, clouds, rain)** | The camera filter relies on Hue-Saturation color palette histograms rather than raw RGB brightness, tolerating warmth/exposure changes without false camera-cut triggers. | Processed normally (`referee_visible = 1`). |
| **Camera cuts to audience, replays, or timeouts** | Color-palette similarity against the main court reference drops sharply; pose estimation is skipped for those frames. | `referee_visible = 0`; angles and `gesture_change` set to `NaN`. |
| **Broadcast cuts to a replay *before* the referee signals** | The model never hallucinates unseen gestures. The missed rally end is recorded as a broadcast coverage limitation during evaluation. | Counted as a False Negative against true rally-end timestamps. |
| **Referee occluded or low keypoint confidence** | If shoulder/elbow/wrist confidence falls below threshold, the frame is marked unavailable rather than passing noisy coordinates to the classifier. | `referee_visible = 0` (features remain `NaN`). |
| **Non-signal movements (wiping sweat, adjusting cap, holding pole)** | Brief or low-amplitude hand movements are filtered out by temporal aggregation over the 2–5 second window and the trained ML classifier. | `referee_signal = 0`. |

---

## 5. Key Risks & Limitations

* **Broadcast Coverage Ceiling:** In YouTube broadcasts, directors sometimes cut to player celebrations or instant replays before the referee raises their arm. This creates an unavoidable upper bound on maximum achievable recall for any referee-based method.
* **Inherent Reaction Lag:** Human referees blow the whistle and signal $1\text{--}3\text{ seconds}$ *after* the ball hits the sand. Downstream users must account for this offset when jumping to rally endings.
* **Camera Panning or Zooming:** While the main camera is mostly static, occasional zoom shifts can move the referee stand partially outside the initial ROI box, requiring a wider margin around the stand or per-segment ROI adjustment.
* **Severe Class Imbalance:** Actual referee signals occupy roughly $5\%$ of total match duration, requiring class weighting or threshold tuning during ML training.

---

## 6. Folder Structure & Setup

```text
referee_detection/
├── README.md                  # Module documentation (this file)
├── scripts/                   # Scripts for ROI setup, YOLO-pose extraction, and ML training
    └── extract_referee_poses.py
│weights/                      # Local storage for .pt and classifier weights (ignored by Git)
└── tests/                     # Notebooks and visual verification tests
```