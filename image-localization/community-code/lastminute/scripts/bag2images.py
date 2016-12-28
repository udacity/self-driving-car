#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('py_save_images')
import sys
import rospy
import cv2
import geo
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import NavSatFix
from cv_bridge import CvBridge, CvBridgeError

from pexif import JpegFile

lat=None
lon=None

#If you want to reduce the number of saved images
save_every_second=True

class image_converter:
    counter=0
    ref_lat = None
    ref_lon = None
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/center_camera/image_color/compressed", CompressedImage, self.img_callbak)
        self.gps_sub = rospy.Subscriber("/vehicle/gps/fix", NavSatFix, self.gps_callbak)

        self.f = open('/tmp/poses.txt', 'w')
        self.reflla = open('/tmp/reflla.txt', 'w')
        self.lla = open('/tmp/lla.txt', 'w')
        if rospy.has_param('~ref_latlon'):
            ref_latlon = rospy.get_param('~ref_latlon')
            self.ref_lat = ref_latlon['lat']
            self.ref_lon = ref_latlon['lon']

    def gps_callbak(self,navsat):
        global lat,lon
        lat = navsat.latitude
        lon = navsat.longitude

    def img_callbak(self,data):
        global lat,lon,save_every_second
        if not lat is None and not lon is None:
            print(lat,lon)
            try:
                # cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                np_arr = np.fromstring(data.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except CvBridgeError as e:
                print(e)
            try:
                cv2.imwrite(filename='/tmp/%08d.jpg'%self.counter, img=cv_image)
                ef = JpegFile.fromFile('/tmp/%08d.jpg'%self.counter)
                ef.set_geo(lat,lon)

                if self.counter == 0 and (not self.ref_lat) and (not self.ref_lon):
                    self.ref_lat = lat
                    self.ref_lon = lon
                    self.reflla.write(str(lat) + "," + str(lon))

                x, y, z = geo.topocentric_from_lla(lat, lon, 0,
                    self.ref_lat, self.ref_lon, 0)
                self.f.write(str(data.header.stamp.to_sec()) + ',' + str(x) + "," + str(y) + ',' + str(z) + '\n')
                self.lla.write('%08d.jpg'%self.counter+','+str(lat)+','+str(lon)+',')
                self.lla.flush()

                if save_every_second==True:
                    lat = None
                    lon = None

            except IOError:
                type, value, traceback = sys.exc_info()
                print >> sys.stderr, "Error opening file:", value
            except JpegFile.InvalidFile:
                type, value, traceback = sys.exc_info()
                print >> sys.stderr, "Error opening file:", value
            try:
                ef.writeFile('/tmp/%08d.jpg'%self.counter)
            except IOError:
                type, value, traceback = sys.exc_info()
                print >> sys.stderr, "Error saving file:", value
            self.counter=self.counter+1


def main(args):
    ic = image_converter()
    rospy.init_node('image_sub', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        self.reflla.close()
        self.lla.close()
        if ic.f:
	    ic.f.close()

if __name__ == '__main__':
    main(sys.argv)
