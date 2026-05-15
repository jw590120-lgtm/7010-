# Demo Script (2 minutes)

| Time        | Action                      | Expected State        | Audio                                 |
| ----------- | --------------------------- | --------------------- | ------------------------------------- |
| 0:00 – 0:30 | Sit upright, look at screen | `focused`             | (silent)                              |
| 0:30 – 1:00 | Turn head 45° left/right    | `distracted` after 4s | "Hey, let's get back to work."        |
| 1:00 – 1:30 | Look down, close eyes       | `drowsy` after 2s     | "You look tired. Take a short break." |
| 1:30 – 2:00 | Walk out of frame           | `absent` after 3s     | (silent — by design)                  |

## Talking Points

1. **Open `rqt_graph`** — show the clean 5-node pipeline.
2. **`rostopic echo /attention/state`** in a side terminal — show
   state transitions happening in real time.
3. **Point out the silence during "absent"** — the robot deliberately
   does not nag someone who has left. This is "selective intervention".
4. **Mention parameter tuning** — `rosparam set /state_judge/ear_thresh 0.18`
   to demonstrate live reconfigurability without rebuilding.