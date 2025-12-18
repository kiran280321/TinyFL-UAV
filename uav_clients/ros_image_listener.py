#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

def callback(msg):
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    cv2.imwrite("latest_frame.jpg", img)

rospy.init_node("uav_camera_listener")
rospy.Subscriber("/uav/camera", Image, callback)
rospy.spin()
