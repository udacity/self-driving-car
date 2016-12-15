import threading
import numpy as np
import rospy
from dbw_mkz_msgs.msg import SteeringCmd
from sensor_msgs.msg import Image


class SteeringNode(object):
    def __init__(self, get_model_callback, model_callback):
        rospy.init_node('steering_model')
        self.model = get_model_callback()
        self.get_model = get_model_callback
        self.predict = model_callback
        self.img =  None
        self.steering = 0.
        self.image_lock = threading.RLock()
        self.image_sub = rospy.Subscriber('/center_camera/image_color', Image,
                                          self.update_image)
        self.pub = rospy.Publisher('/vehicle/steering_cmd',
                                   SteeringCmd, queue_size=1)
        rospy.Timer(rospy.Duration(.02), self.get_steering)

    def update_image(self, img):
        d = map(ord, img.data)
        arr = np.ndarray(shape=(img.height, img.width, 3),
                         dtype=np.int,
                         buffer=np.array(d))[:,:,::-1]
        if self.image_lock.acquire(True):
            self.img = arr
            if self.model is None:
                self.model = self.get_model()
            self.steering = self.predict(self.model, self.img)
            self.image_lock.release()

    def get_steering(self, event):
        if self.img is None:
            return
        message = SteeringCmd()
        message.enable = True
        message.ignore = False
        message.steering_wheel_angle_cmd = self.steering
        self.pub.publish(message)
