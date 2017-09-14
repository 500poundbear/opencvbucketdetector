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
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/usb_cam_front/image_raw/compressed', CompressedImage, self.callback)
	
	def callback(self,data):
		cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
		hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)


		upper_red = np.array([220, 150, 150])
		lower_red= np.array([0, 0, 0])
		
		mask = cv2.inRange(hsv, lower_red, upper_red)
		kernel = np.ones((5,5), np.uint8)
		
		mask = cv2.erode(mask, kernel, iterations = 1)
		mask = cv2.dilate(mask, kernel, iterations = 1)
		
		res = cv2.bitwise_and(hsv, hsv, mask=mask)

		im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		largestArea = 0
		largestRect = None
		for c in cnts:
			if cv2.contourArea(c) < 100:
				continue
			if cv2.contourArea(c) > largestArea:
				largestArea = cv2.contourArea(c)
				largestRect = cv2.boundingRect(c)
		if largestArea > 0:
			(x, y, w, h) = largestRect
			cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		cv2.imshow("Bucket Detector 2", cv_image)
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
