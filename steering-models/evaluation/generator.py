
import rosbag
from StringIO import StringIO
from scipy import misc
import numpy as np

KEY_NAME = {
    '/vehicle/steering_report': 'steering',
    '/center_camera/image_color/c': 'image',
}

def update(msg, d):
    key = KEY_NAME.get(msg.topic)
    if key is None: return
    d[key] = msg


def gen(bag):
    print 'Getting bag'
    bag = rosbag.Bag(bag)
    print 'Got bag'
    
    image = {}
    total = bag.get_message_count()
    count = 0
    for e in bag.read_messages():
        count += 1
        if count % 10000 == 0:
            print count, '/', total
        if e.topic in ['/center_camera/image_color/compressed']:
            if len({'steering'} - set(image.keys())):
                continue
            if image['steering'].message.speed < 5.: continue
            s = StringIO(e.message.data)
            img = misc.imread(s)
            yield img, np.copy(img), image['steering'].message.speed,\
                  image['steering'].message.steering_wheel_angle, e.timestamp.to_nsec()
            last_ts = e.timestamp.to_nsec()
        else:
            update(e, image)
