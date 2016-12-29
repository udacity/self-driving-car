#!/usr/bin/env python
"""
Usage: ./mergeBag.py -a bag1.bag -b bag2.bag -o outputBag.bag

Connects two bags right after one another. Meaning that if bag1 starts at time 10 and
end at time 12 and bag2 starts at time 12 and end at time 15, then the resulting 
outputBag will start at time 10 and ends at time 15.
"""

import rosbag
import rospy
from optparse import OptionParser
import sys
import argparse
import os.path

parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')
parser.add_argument('-a','--input1', help='Input file name',required=True)
parser.add_argument('-b','--input2', help='Input file name',required=True)
parser.add_argument('-o','--output',help='Output file name', required=True)
args = parser.parse_args()

if not os.path.isfile(args.input1):
    print "Invalid input file " +  args.input
    exit() 

with rosbag.Bag(args.output, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(args.input1).read_messages():
            outbag.write(topic, msg, t)
    for topic, msg, t in rosbag.Bag(args.input2).read_messages():
            outbag.write(topic, msg, t)
