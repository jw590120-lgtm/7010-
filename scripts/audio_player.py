#!/usr/bin/env python3
"""
audio_player.py
Receives alert text and plays it via sound_play TTS.
"""
import rospy
from sound_play.libsoundplay import SoundClient
from std_msgs.msg import String


class AudioPlayer:
    def __init__(self):
        rospy.init_node('audio_player')
        self.client = SoundClient(blocking=False)
        rospy.sleep(1.0)
        rospy.Subscriber('/attention/alert',
                         String, self.cb, queue_size=10)
        rospy.loginfo("audio_player started")

    def cb(self, msg):
        rospy.loginfo(f"speak: {msg.data}")
        self.client.say(msg.data)


if __name__ == '__main__':
    try:
        AudioPlayer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass