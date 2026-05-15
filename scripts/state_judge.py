#!/usr/bin/env python3
"""
state_judge.py
Rule-based state machine. Requires a candidate state to be sustained
for N seconds before transitioning.
"""
import rospy
from focus_companion.msg import AttentionFeatures, AttentionState


class StateJudge:
    def __init__(self):
        rospy.init_node('state_judge')
        self.ear_thresh = rospy.get_param('/state_judge/ear_thresh', 0.21)
        self.pitch_down_thresh = rospy.get_param(
            '/state_judge/pitch_down_thresh', 20.0)
        self.yaw_thresh = rospy.get_param('/state_judge/yaw_thresh', 25.0)
        self.absent_sec = rospy.get_param('/state_judge/absent_sec', 3.0)
        self.drowsy_sec = rospy.get_param('/state_judge/drowsy_sec', 2.0)
        self.distract_sec = rospy.get_param('/state_judge/distract_sec', 4.0)

        self.state = 'focused'
        self.state_start = rospy.Time.now()
        self.cand_since = {'absent': None, 'drowsy': None, 'distracted': None}

        self.pub = rospy.Publisher('/attention/state',
                                   AttentionState, queue_size=10)
        rospy.Subscriber('/attention/features_smooth',
                         AttentionFeatures, self.cb, queue_size=20)
        rospy.loginfo("state_judge started")

    def _update_candidate(self, name, condition_met, now):
        if condition_met:
            if self.cand_since[name] is None:
                self.cand_since[name] = now
        else:
            self.cand_since[name] = None

    def cb(self, msg):
        now = rospy.Time.now()

        absent_cond = not msg.face_detected
        drowsy_cond = msg.face_detected and (
            msg.eye_aspect_ratio < self.ear_thresh
            or msg.head_pitch < -self.pitch_down_thresh
        )
        distract_cond = (
            msg.face_detected
            and not drowsy_cond
            and abs(msg.head_yaw) > self.yaw_thresh
        )

        self._update_candidate('absent', absent_cond, now)
        self._update_candidate('drowsy', drowsy_cond, now)
        self._update_candidate('distracted', distract_cond, now)

        new_state = 'focused'
        for name, need in [('absent', self.absent_sec),
                           ('drowsy', self.drowsy_sec),
                           ('distracted', self.distract_sec)]:
            t0 = self.cand_since[name]
            if t0 is not None and (now - t0).to_sec() >= need:
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
        StateJudge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass