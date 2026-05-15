# Focus Companion

A ROS1 (Noetic) based attention monitoring robot for Jupiter Robot Juno 2.
Detects user attention states (focused / distracted / drowsy / absent) via
camera and provides selective audio feedback.

## Architecturecamera → attention_detect → attention_smoother → state_judge
↓
audio_player ← response_manager

## Requirements

- Ubuntu 20.04
- ROS Noetic
- Python 3.8
- Jupiter Robot Juno 2 (or any RGB camera publishing `sensor_msgs/Image`)

## Installation

```bashcd ~/catkin_ws/src
git clone https://github.com/<your-username>/focus_companion.git
cd ~/catkin_ws
catkin_make
source devel/setup.bashInstall Python dependencies
pip3 install --user -r src/focus_companion/requirements.txtInstall ROS dependencies
sudo apt install -y ros-noetic-sound-play ros-noetic-cv-bridge

## Usage

```bashTerminal 1: start camera (Jupiter Robot)
roslaunch jupiterobot2_bringup minimal.launchTerminal 2: start Focus Companion
roslaunch focus_companion focus_companion.launchTerminal 3 (optional): monitor state
rostopic echo /attention/state

## Configuration

Tune thresholds in `config/params.yaml`:

| Param | Default | Meaning |
|---|---|---|
| `ear_thresh` | 0.21 | Eye Aspect Ratio threshold for drowsy detection |
| `pitch_down_thresh` | 20.0 | Head pitch (degrees) for looking down |
| `yaw_thresh` | 25.0 | Head yaw (degrees) for looking sideways |
| `drowsy_sec` | 2.0 | Sustained seconds before declaring drowsy |
| `distract_sec` | 4.0 | Sustained seconds before declaring distracted |
| `absent_sec` | 3.0 | Sustained seconds before declaring absent |
| `cooldown_sec` | 10.0 | Minimum interval between repeated alerts |

## Topics

| Topic | Type | Description |
|---|---|---|
| `/attention/features` | `AttentionFeatures` | Raw per-frame features |
| `/attention/features_smooth` | `AttentionFeatures` | Smoothed features |
| `/attention/state` | `AttentionState` | Final attention state |
| `/attention/alert` | `std_msgs/String` | Audio alert text |

## Authors

Group X - WQF7XXX, Universiti Malaya

## License

MIT
