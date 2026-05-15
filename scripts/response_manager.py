#!/usr/bin/env python3
"""
response_manager.py
Maps state transitions to audio alerts. Implements cooldown to
avoid spamming the user.
"""
import rospy
from focus_companion.msg import AttentionState
from std_msgs.msg import String


class ResponseManager:
    def __init__(self):
        rospy.init_node('response_manager')
        self.cooldown = rospy.get_param(
            '/response_manager/cooldown_sec', 10.0)
        self.alert_distracted = rospy.get_param(
            '/response_manager/alert_distracted',
            "Hey, let's get back to work.")
        self.alert_drowsy = rospy.get_param(
            '/response_manager/alert_drowsy',
            "You look tired. Take a short break.")

        self.alert_map = {
            'distracted': self.alert_distracted,
            'drowsy':     self.alert_drowsy,
            'absent':     "",
            'focused':    "",
        }

        self.last_state = 'focused'
        self.last_alert_time = {}

        self.pub = rospy.Publisher('/attention/alert',
                                   String, queue_size=10)
        rospy.Subscriber('/attention/state',
                         AttentionState, self.cb, queue_size=10)
        rospy.loginfo("response_manager started")

    def cb(self, msg):
        now = rospy.Time.now()
        if msg.state == self.last_state:
            return

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
        ResponseManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass