#!/usr/bin/env python3
"""
attention_detect.py
Subscribes to camera image, extracts face landmarks via MediaPipe,
and publishes raw attention features.
"""
import rospy
import cv2
import numpy as np
import mediapipe as mp
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from focus_companion.msg import AttentionFeatures

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * horiz + 1e-6)


def head_pose(landmarks, w, h):
    image_pts = np.array([
        (landmarks[1].x * w,   landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x * w,  landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w,  landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h),
    ], dtype=np.float64)
    model_pts = np.array([
        (0.0,   0.0,    0.0),
        (0.0,   -63.6,  -12.5),
        (-43.3, 32.7,   -26.0),
        (43.3,  32.7,   -26.0),
        (-28.9, -28.9,  -24.1),
        (28.9,  -28.9,  -24.1),
    ])
    focal = w
    cam_matrix = np.array([[focal, 0, w / 2],
                           [0, focal, h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam_matrix, dist)
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw   = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    return pitch, yaw


class AttentionDetect:
    def __init__(self):
        rospy.init_node('attention_detect')
        self.bridge = CvBridge()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pub = rospy.Publisher('/attention/features',
                                   AttentionFeatures, queue_size=10)
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
            feat.eye_aspect_ratio = float((ear_l + ear_r) / 2.0)
            feat.head_pitch = float(pitch)
            feat.head_yaw = float(yaw)
            feat.face_center_x = float(lm[1].x)
            feat.face_center_y = float(lm[1].y)
        else:
            feat.face_detected = False

        self.pub.publish(feat)


if __name__ == '__main__':
    try:
        AttentionDetect()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass