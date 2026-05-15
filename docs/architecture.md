```markdown
# Architecture

## Node Graph
```

/camera/rgb/image_raw  (sensor_msgs/Image) │ ▼ ┌──────────────────┐ │ attention_detect │  MediaPipe Face Mesh → EAR + head pose └──────────────────┘ │  /attention/features ▼ ┌────────────────────┐ │ attention_smoother │  Sliding window average └────────────────────┘ │  /attention/features_smooth ▼ ┌──────────────┐ │ state_judge  │  Rule-based + sustained-time state machine └──────────────┘ │  /attention/state ▼ ┌────────────────────┐ │ response_manager   │  Transition-triggered + cooldown └────────────────────┘ │  /attention/alert ▼ ┌──────────────┐ │ audio_player │  sound_play TTS └──────────────┘

```
## Design Decisions

### Why split `detect` and `smoother`?
The perception layer should stay stateless and only do feature extraction.
Smoothing is a separate, swappable concern — today it's a simple moving
average, tomorrow it could be a Kalman filter without changing the detector.

### Why split `judge` and `response`?
State estimation and interaction policy are decoupled. Adding a new state
(e.g. "anxious") only requires changes to `state_judge`, not `audio_player`.

### Why sustained-time state transitions?
MediaPipe is noisy. A single frame of looking down should not trigger
"drowsy". Requiring N consecutive seconds of evidence makes the system
robust to brief head movements, blinks, and detection failures.

### Why cooldown in `response_manager`?
A robot that nags every 2 seconds is worse than no robot. Cooldown
enforces "selective intervention" — the system stays silent unless
something meaningful changed.
```