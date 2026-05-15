#!/usr/bin/env python3
"""
attention_smoother.py
Buffers features over a sliding window and publishes averaged values
to reduce MediaPipe per-frame jitter.
"""
import rospy
from collections import deque
from focus_companion.msg import AttentionFeatures


class Smoother:
    def __init__(self):
        rospy.init_node('attention_smoother')
        self.window_size = rospy.get_param(
            '/attention_smoother/window_size', 15
        )
        self.buf = deque(maxlen=self.window_size)
        self.pub = rospy.Publisher('/attention/features_smooth',
                                   AttentionFeatures, queue_size=10)
        rospy.Subscriber('/attention/features',
                         AttentionFeatures, self.cb, queue_size=20)
        rospy.loginfo(f"attention_smoother started, window={self.window_size}")

    def cb(self, msg):
        self.buf.append(msg)
        if len(self.buf) < 3:
            return

        face_rate = sum(1 for m in self.buf if m.face_detected) / len(self.buf)
        detected = [m for m in self.buf if m.face_detected]

        out = AttentionFeatures()
        out.header.stamp = rospy.Time.now()
        out.face_detected = face_rate > 0.5

        if detected:
            n = len(detected)
            out.eye_aspect_ratio = sum(m.eye_aspect_ratio for m in detected) / n
            out.head_pitch       = sum(m.head_pitch       for m in detected) / n
            out.head_yaw         = sum(m.head_yaw         for m in detected) / n
            out.face_center_x    = sum(m.face_center_x    for m in detected) / n
            out.face_center_y    = sum(m.face_center_y    for m in detected) / n

        self.pub.publish(out)


if __name__ == '__main__':
    try:
        Smoother()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass