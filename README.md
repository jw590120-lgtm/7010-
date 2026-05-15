# 7010-

好的，思路很清晰——五个节点、规则判断、不训练模型，这套方案对 ROS 集成评分来说是合适的体量。下面给你一个能直接跑起来的完整骨架。

## 整体规划

数据流：
```
/camera/rgb/image_raw 
   → attention_detect → /attention/features (10~30Hz, 有噪声)
   → attention_smoother → /attention/features_smooth (平滑后)
   → state_judge → /attention/state (focused/distracted/drowsy/absent)
   → response_manager → /attention/alert (只在状态切换时触发)
   → audio_player → sound_play
```

关键设计点：smoother 解决 MediaPipe 抖动，state_judge 用"连续 N 秒满足条件"来防误判，response_manager 用 cooldown 防止机器人话痨。

---

## Step 1 - 环境准备

```bash
# 1. 装 MediaPipe 和 sound_play
pip3 install mediapipe opencv-python
sudo apt install ros-noetic-sound-play ros-noetic-cv-bridge

# 2. 确认摄像头话题（Juno 2 上确认一下）
roslaunch jupiterobot2_bringup jupiterobot2_bringup.launch  # 或你们平时用的
rostopic list | grep image
# 假设是 /camera/rgb/image_raw，后面以这个为准
```

## Step 2 - 创建包

```bash
cd ~/catkin_ws/src
catkin_create_pkg focus_companion rospy std_msgs sensor_msgs cv_bridge message_generation message_runtime sound_play

cd focus_companion
mkdir -p scripts launch msg audio
```

**`msg/AttentionFeatures.msg`** — 原始特征
```
Header header
bool face_detected
float32 eye_aspect_ratio
float32 head_pitch
float32 head_yaw
float32 face_center_x
float32 face_center_y
```

**`msg/AttentionState.msg`** — 状态
```
Header header
string state
float32 duration
```

**`CMakeLists.txt`** 关键改动（找到对应位置取消注释/添加）：
```cmake
find_package(catkin REQUIRED COMPONENTS
  rospy std_msgs sensor_msgs cv_bridge
  message_generation sound_play
)

add_message_files(
  FILES
  AttentionFeatures.msg
  AttentionState.msg
)

generate_messages(
  DEPENDENCIES std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge message_runtime sound_play
)
```

**`package.xml`** 加上：
```xml
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```

然后编译：
```bash
cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

---

## Step 3 - 五个节点

### Node 1: `scripts/attention_detect.py`

```python
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from focus_companion.msg import AttentionFeatures

# MediaPipe Face Mesh 眼部关键点索引
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)

def head_pose(landmarks, w, h):
    # 用 6 个关键点 + solvePnP 估计 pitch/yaw
    image_pts = np.array([
        (landmarks[1].x * w,   landmarks[1].y * h),    # 鼻尖
        (landmarks[152].x * w, landmarks[152].y * h),  # 下巴
        (landmarks[33].x * w,  landmarks[33].y * h),   # 左眼外角
        (landmarks[263].x * w, landmarks[263].y * h),  # 右眼外角
        (landmarks[61].x * w,  landmarks[61].y * h),   # 嘴左
        (landmarks[291].x * w, landmarks[291].y * h),  # 嘴右
    ], dtype=np.float64)
    model_pts = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ])
    focal = w
    cam_matrix = np.array([[focal, 0, w/2],
                           [0, focal, h/2],
                           [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(model_pts, image_pts, cam_matrix, dist)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
    yaw   = np.degrees(np.arctan2(rmat[1,0], rmat[0,0]))
    return pitch, yaw

class AttentionDetect:
    def __init__(self):
        rospy.init_node('attention_detect')
        self.bridge = CvBridge()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.pub = rospy.Publisher('/attention/features', AttentionFeatures, queue_size=10)
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.cb, queue_size=1)
        rospy.loginfo("attention_detect node started")

    def cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn(f"cv_bridge error: {e}")
            return
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        feat = AttentionFeatures()
        feat.header.stamp = rospy.Time.now()

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            pitch, yaw = head_pose(lm, w, h)

            feat.face_detected = True
            feat.eye_aspect_ratio = (ear_l + ear_r) / 2.0
            feat.head_pitch = float(pitch)
            feat.head_yaw   = float(yaw)
            feat.face_center_x = lm[1].x  # 归一化坐标
            feat.face_center_y = lm[1].y
        else:
            feat.face_detected = False

        self.pub.publish(feat)

if __name__ == '__main__':
    try:
        AttentionDetect()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

### Node 2: `scripts/attention_smoother.py`

```python
#!/usr/bin/env python3
import rospy
from collections import deque
from focus_companion.msg import AttentionFeatures

class Smoother:
    def __init__(self):
        rospy.init_node('attention_smoother')
        self.window_size = rospy.get_param('~window_size', 15)  # ~0.5s @ 30Hz
        self.buf = deque(maxlen=self.window_size)
        self.pub = rospy.Publisher('/attention/features_smooth', AttentionFeatures, queue_size=10)
        rospy.Subscriber('/attention/features', AttentionFeatures, self.cb, queue_size=20)
        rospy.loginfo(f"attention_smoother started, window={self.window_size}")

    def cb(self, msg):
        self.buf.append(msg)
        if len(self.buf) < 3:
            return

        # 人脸检出率：窗口内 face_detected=True 的比例
        face_rate = sum(1 for m in self.buf if m.face_detected) / len(self.buf)
        detected_msgs = [m for m in self.buf if m.face_detected]

        out = AttentionFeatures()
        out.header.stamp = rospy.Time.now()
        out.face_detected = face_rate > 0.5

        if detected_msgs:
            n = len(detected_msgs)
            out.eye_aspect_ratio = sum(m.eye_aspect_ratio for m in detected_msgs) / n
            out.head_pitch       = sum(m.head_pitch for m in detected_msgs) / n
            out.head_yaw         = sum(m.head_yaw for m in detected_msgs) / n
            out.face_center_x    = sum(m.face_center_x for m in detected_msgs) / n
            out.face_center_y    = sum(m.face_center_y for m in detected_msgs) / n
        self.pub.publish(out)

if __name__ == '__main__':
    try:
        Smoother(); rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

### Node 3: `scripts/state_judge.py`

核心规则：**单帧不算数，必须持续 N 秒满足条件才切换状态**。这是为什么需要 smoother + judge 两层——前者去抖，后者做时间累积。

```python
#!/usr/bin/env python3
import rospy
from focus_companion.msg import AttentionFeatures, AttentionState

class StateJudge:
    def __init__(self):
        rospy.init_node('state_judge')
        # 阈值，全部可调参
        self.ear_thresh       = rospy.get_param('~ear_thresh', 0.21)
        self.pitch_down_thresh = rospy.get_param('~pitch_down_thresh', 20.0)
        self.yaw_thresh        = rospy.get_param('~yaw_thresh', 25.0)
        self.absent_sec        = rospy.get_param('~absent_sec', 3.0)
        self.drowsy_sec        = rospy.get_param('~drowsy_sec', 2.0)
        self.distract_sec      = rospy.get_param('~distract_sec', 4.0)

        self.state = 'focused'
        self.state_start = rospy.Time.now()

        # 每个候选状态的持续计时
        self.cand_since = {'absent': None, 'drowsy': None, 'distracted': None}

        self.pub = rospy.Publisher('/attention/state', AttentionState, queue_size=10)
        rospy.Subscriber('/attention/features_smooth', AttentionFeatures, self.cb, queue_size=20)
        rospy.loginfo("state_judge started")

    def update_candidate(self, name, condition_met, now):
        if condition_met:
            if self.cand_since[name] is None:
                self.cand_since[name] = now
        else:
            self.cand_since[name] = None

    def cb(self, msg):
        now = rospy.Time.now()

        absent_cond   = not msg.face_detected
        drowsy_cond   = msg.face_detected and (
            msg.eye_aspect_ratio < self.ear_thresh or msg.head_pitch < -self.pitch_down_thresh
        )
        distract_cond = msg.face_detected and (not drowsy_cond) and (
            abs(msg.head_yaw) > self.yaw_thresh
        )

        self.update_candidate('absent', absent_cond, now)
        self.update_candidate('drowsy', drowsy_cond, now)
        self.update_candidate('distracted', distract_cond, now)

        # 按优先级判定（absent > drowsy > distracted > focused）
        new_state = 'focused'
        for name, need_sec in [('absent', self.absent_sec),
                               ('drowsy', self.drowsy_sec),
                               ('distracted', self.distract_sec)]:
            t0 = self.cand_since[name]
            if t0 is not None and (now - t0).to_sec() >= need_sec:
                new_state = name
                break

        if new_state != self.state:
            rospy.loginfo(f"state: {self.state} -> {new_state}")
            self.state = new_state
            self.state_start = now

        out = AttentionState()
        out.header.stamp = now
        out.state = self.state
        out.duration = (now - self.state_start).to_sec()
        self.pub.publish(out)

if __name__ == '__main__':
    try:
        StateJudge(); rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

### Node 4: `scripts/response_manager.py`

只在**状态切换瞬间**发提示，并且加 cooldown 防止反复触发。这就是你说的"选择性介入"。

```python
#!/usr/bin/env python3
import rospy
from focus_companion.msg import AttentionState
from std_msgs.msg import String

class ResponseManager:
    def __init__(self):
        rospy.init_node('response_manager')
        self.cooldown = rospy.get_param('~cooldown_sec', 10.0)
        self.last_state = 'focused'
        self.last_alert_time = {}

        # 状态 → 提示语映射
        self.alert_map = {
            'distracted': "Hey, let's get back to work.",
            'drowsy':     "You look tired. Take a short break.",
            'absent':     "",  # 离开不打扰
            'focused':    ""   # 进入专注也不需要播
        }

        self.pub = rospy.Publisher('/attention/alert', String, queue_size=10)
        rospy.Subscriber('/attention/state', AttentionState, self.cb, queue_size=10)
        rospy.loginfo("response_manager started")

    def cb(self, msg):
        now = rospy.Time.now()
        if msg.state == self.last_state:
            return  # 只在状态切换时触发

        rospy.loginfo(f"transition: {self.last_state} -> {msg.state}")
        self.last_state = msg.state

        text = self.alert_map.get(msg.state, "")
        if not text:
            return

        last_t = self.last_alert_time.get(msg.state)
        if last_t and (now - last_t).to_sec() < self.cooldown:
            rospy.loginfo(f"{msg.state} on cooldown, skip")
            return

        self.last_alert_time[msg.state] = now
        self.pub.publish(String(data=text))

if __name__ == '__main__':
    try:
        ResponseManager(); rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

### Node 5: `scripts/audio_player.py`

直接用 `sound_play` 的 TTS，省去你录音频文件的麻烦。

```python
#!/usr/bin/env python3
import rospy
from sound_play.libsoundplay import SoundClient
from std_msgs.msg import String

class AudioPlayer:
    def __init__(self):
        rospy.init_node('audio_player')
        self.client = SoundClient(blocking=False)
        rospy.sleep(1.0)  # 等 sound_play server 起来
        rospy.Subscriber('/attention/alert', String, self.cb, queue_size=10)
        rospy.loginfo("audio_player started")

    def cb(self, msg):
        rospy.loginfo(f"speak: {msg.data}")
        self.client.say(msg.data)

if __name__ == '__main__':
    try:
        AudioPlayer(); rospy.spin()
    except rospy.ROSInterruptException:
        pass
```

记得给所有 py 文件加可执行权限：
```bash
chmod +x ~/catkin_ws/src/focus_companion/scripts/*.py
```

---

## Step 4 - Launch 文件

**`launch/focus_companion.launch`**

```xml
<launch>
  <!-- 摄像头话题，根据 Juno 2 实际话题改 -->
  <arg name="image_topic" default="/camera/rgb/image_raw"/>

  <!-- sound_play server -->
  <node name="soundplay_node" pkg="sound_play" type="soundplay_node.py" output="screen"/>

  <node name="attention_detect" pkg="focus_companion" type="attention_detect.py" output="screen">
    <remap from="/camera/rgb/image_raw" to="$(arg image_topic)"/>
  </node>

  <node name="attention_smoother" pkg="focus_companion" type="attention_smoother.py" output="screen">
    <param name="window_size" value="15"/>
  </node>

  <node name="state_judge" pkg="focus_companion" type="state_judge.py" output="screen">
    <param name="ear_thresh" value="0.21"/>
    <param name="drowsy_sec" value="2.0"/>
    <param name="distract_sec" value="4.0"/>
    <param name="absent_sec" value="3.0"/>
  </node>

  <node name="response_manager" pkg="focus_companion" type="response_manager.py" output="screen">
    <param name="cooldown_sec" value="10.0"/>
  </node>

  <node name="audio_player" pkg="focus_companion" type="audio_player.py" output="screen"/>
</launch>
```

---

## Step 5 - 跑起来 & Demo

```bash
# Terminal 1: 启动 Juno 2 摄像头（你们平时怎么起就怎么起）
roslaunch jupiterobot2_bringup minimal.launch  # 或 jupiterobot2_vision_mediapipe 那个

# Terminal 2: 启动我们的系统
roslaunch focus_companion focus_companion.launch

# Terminal 3: 调试 - 观察状态变化
rostopic echo /attention/state
# 或看完整数据流
rqt_graph
```

**Demo 脚本（2 分钟）**：
1. 0:00–0:30 正脸看屏幕 → 状态稳定在 `focused`，无播报 ✅
2. 0:30–1:00 头转左/右 45° → 持续 4 秒后切到 `distracted` → 播一句 "Hey, let's get back to work." ✅
3. 1:00–1:30 低头闭眼装睡 → 持续 2 秒后切到 `drowsy` → 播一句 "You look tired..." ✅
4. 1:30–2:00 走出画面 → 持续 3 秒后切到 `absent` → 静默（这个点要主动讲给评分老师，体现"选择性介入"）✅

---

## 一些容易踩的坑

**话题名不对**：Juno 2 的 RGB 相机话题不一定是 `/camera/rgb/image_raw`，先 `rostopic list` 确认，在 launch 里 remap。

**sound_play 没声音**：Ubuntu 默认输出可能到 HDMI 了，`pavucontrol` 切到正确输出设备；测试用 `rosrun sound_play say.py "hello"`。

**MediaPipe 初始化慢**：第一帧推理需要几秒，启动后等一下再开始 demo。

**EAR 阈值因人而异**：0.21 是经验值，眼睛细的同学（包括很多亚洲面孔）平时 EAR 就在 0.2 上下，跑一次记录正常状态的 EAR，再把阈值调到正常值 - 0.05 左右。建议 demo 前用 `rostopic echo /attention/features_smooth` 实时看一遍数据，再校准阈值。

**演示时被自己说话干扰**：sound_play 输出会被摄像头那边的人看到张嘴 → 影响 face landmark。无所谓，状态判定看的是 EAR 和头部姿态，不看嘴。

---

## 给老师讲解时的几个加分点

ROS 集成层面老师想看到的是节点解耦清晰、话题设计合理、参数化、launch 一键起。这套架构里可以专门强调：
- **为什么拆 detect 和 smoother**：感知层只做特征提取无状态化，平滑层独立可替换（卡尔曼/滑窗都能换）
- **为什么拆 judge 和 response**：状态机和交互策略解耦，老师以后想加新状态（比如"焦虑"）只改 judge，不动 audio
- **custom msg 而不是 std_msgs**：体现对 ROS 消息系统的理解
- **rosparam 暴露阈值**：不用改代码就能调，符合工程实践

如果你想再多加一点料，可以加个 `rqt_plot` 截图展示 EAR/pitch/yaw 的时序曲线，配合状态切换时刻——视觉上很有说服力，对评分非常友好。

要不要我帮你把这个项目的 README（中英双语）也写一版？答辩或者 GitHub 提交都能直接用。
