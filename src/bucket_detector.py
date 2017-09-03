#!/usr/bin/env python
import roslib
roslib.load_manifest('bucket_detector')
import sys
import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:
	def __init__(self):
		self.image_pub = rospy.Publisher("/bucket_detector/namnam", Image, queue_size=100)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/usb_cam_front/image_raw/compressed', CompressedImage, self.callback)
	
	def callback(self,data):
		cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
		hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

		lower_red= np.array([30, 150, 50])
		upper_red = np.array([255, 255, 180])
		
		mask = cv2.inRange(hsv, lower_red, upper_red)
		
		res = cv2.bitwise_and(cv_image, cv_image, mask=mask)
		modified_image = res
		cv2.imshow("Bucket Detector", modified_image)
		cv2.waitKey(3)

def main(args):
	ic = image_converter()
	rospy.init_node('bucket_detector', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shut down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
