#!/usr/bin/env python

import sys
import subprocess
import os
import signal

import tf
import rospy
import roslib
import math

from std_msgs.msg import Float64

FNULL = open(os.devnull, 'w')

bags = [
	("/home/robo/bag/Sunny.bag", "/home/robo/bag/Sunny.bag"),
	("/home/robo/bag/Overcast.bag", "/home/robo/bag/Overcast.bag"),
	("/home/robo/bag/Highway.bag", "/home/robo/bag/Highway.bag")]

mses = []

rospy.init_node('validator', anonymous=True)

slam = roslib.packages.find_node('slam', 'slam')
rosbagPlay = roslib.packages.find_node('rosbag', 'play')
rospub = roslib.packages.find_node('rostopic', 'rostopic')
validator = roslib.packages.find_node('validator', 'validator')

if not rosbagPlay:
	print("Cannot find rosbag/play executable")
	sys.exit(-1);

if not slam:
	print("Cannot find slam executable")
	sys.exit(-1);


if not rospub:
	print("Cannot find rospub executable")
	sys.exit(-1);

cmdLocalize = [rospub[0],'pub', "/localize", "std_msgs/Empty", "-1"]

mse = 0.0

def callback(data):
	global mse;
	mse = data.data

rospy.Subscriber("/validator/mse", Float64, callback)

devPlay = ["-d", "7"]
try:
	for a, b in bags:
		mse = -1.0
		print a,": ";
		slamRun = subprocess.Popen([slam[0]], stdout=FNULL, stderr=subprocess.STDOUT);
		#learning phase
		cmd = [rosbagPlay[0], '--rate', "6.0", '--clock',str(a)]
		rosbag = subprocess.Popen(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
		rosbag.wait()

		# localization phase
		subprocess.call(cmdLocalize)
		validatorRun = subprocess.Popen([validator[0]], stdout=FNULL, stderr=subprocess.STDOUT)

		cmd = [rosbagPlay[0], '--rate', "1.5", '--clock',str(b)]

		rosbag = subprocess.Popen(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
		rosbag.wait()

		os.kill(slamRun.pid, signal.SIGINT)
		os.kill(validatorRun.pid, signal.SIGINT)
		slamRun.wait()
		validatorRun.wait()

		mses.append((b,mse))
except:
	pass

print(mses)

with open("/home/robo/validation.txt", "a") as myfile:
    myfile.write(str(mses))
    myfile.write("\n")

