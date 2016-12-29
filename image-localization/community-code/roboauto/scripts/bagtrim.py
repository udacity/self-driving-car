#!/usr/bin/env python
import sys
import roslib; roslib.load_manifest('bagedit')
import rospy
import rosbag
from rospy import rostime
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'bagtrim.py',
        description='Trims the beginning and the and of a bagfile.')
    parser.add_argument('-s', type=float, help='start time in seconds', 
        default = None, metavar = "start_time")
    parser.add_argument('-e', type=float, help='end time in seconds', 
        default = None, metavar = "end_time")
    parser.add_argument('-o', type=str, help='name of the output file', 
        default = None, metavar = "output_file")
    parser.add_argument('-a', help='use absolute timestamps', 
        default = False, action="store_true")
    parser.add_argument('bagfile', type=str, help='path to a bagfile')
    args = parser.parse_args()
    return args


def trim_bag(bagfile, start_time, end_time, outfile = None, 
        absolute_timestamps = False):
    #get min and max time in bagfile
    limits = get_limits(bagfile)
    start_duration = limits[0] - rostime.Time.from_sec(0)
    #check output file
    if outfile == None:
        pattern = bagfile + "_cut_%i.bag"
        outfile = bagfile + "_cut.bag"
        index = 0
        while (os.path.exists(outfile)):
            outfile = pattern%index
            index += 1
    #create start time
    if start_time == None:
        start_time = limits[0]
    else:
        start_time = rostime.Time.from_sec(start_time)
        if not absolute_timestamps:
            start_time = start_time + start_duration
    #create end time
    if end_time == None:
        end_time = limits[1]
    else:
        end_time = rostime.Time.from_sec(end_time)
        if not absolute_timestamps:
            end_time = end_time + start_duration
    #check times to prevent user mistakes
    if  end_time < start_time:
        raise SystemExit("End time (%f) is lower than start time (%f)"%\
            (start_time.to_sec(), end_time.to_sec()))
    #output some information
    print "cut from %fs to %fs"%(
        (start_time - limits[0]).to_sec(),
        (end_time - limits[0]).to_sec())
    print "writing to %s."%outfile
    #copy bagfile
    outbag = rosbag.Bag(outfile, 'w')
    try:
        for topic, msg, t in rosbag.Bag(bagfile).read_messages(
                start_time = start_time,
                end_time = end_time):
            outbag.write(topic, msg, t)
    finally:
        outbag.close()

def get_limits(bagfile):
    end_time = None
    start_time = None
    for topic, msg, t in rosbag.Bag(bagfile).read_messages():
        if start_time == None or t < start_time:
            start_time = t
        if end_time == None or t > end_time:
            end_time = t
    return (start_time, end_time)
    
if __name__ == "__main__":
    args = parse_args()
    if args.s == None and args.e == None:
        limits = get_limits(args.bagfile)
        print "length of %s: %f seconds"%\
            (args.bagfile, (limits[1] - limits[0]).to_sec())
        raise SystemExit()
    trim_bag(args.bagfile, 
        start_time = args.s, 
        end_time = args.e, 
        outfile = args.o, 
        absolute_timestamps = args.a)
