#!/usr/bin/python
import sys, math
import rosbag

#topics that will remain in output bags
allowed_topics = ["/center_camera/image_color/compressed", "/center_camera/image_color",
					"/vehicle/gps/fix"]

def getDigitCount(number):
	"""How many digits will have the highest numbered file? e.g. if there will be 25
	resulting bags the result is 2, for 300 bags the result is 3, for 8 bags it is 1"""
	return len(str(number))

def getName(path):
	"""removes ending from path ('/home/robo/Overcast.bag' -> '/home/robo/Overcast')"""
	parts = path.split('.')
	if len(parts) > 1:
		del parts[-1]
	return ".".join(parts)

def splitBag(path, part_duration):
	bag = rosbag.Bag(path, 'r')
	start = int(math.floor(bag.get_start_time()))
	end = bag.get_end_time()
	total_duration = end - start
	part_count = int(math.ceil(total_duration / part_duration)) #how many bags will be produced


	digitCount = getDigitCount(part_count + 1)
	opened_bags = []
	name = getName(path)
	for i in range(part_count):
		opened_bags.append(rosbag.Bag(name + str(i + 1).zfill(digitCount) + ".bag", 'w'))

	last_percentage = -5;
	for topic, msg, t in bag.read_messages(topics=allowed_topics):
		index = int((msg.header.stamp.secs - start) / part_duration)
		opened_bags[index].write(topic, msg, t=t)
		percentage = int(((msg.header.stamp.secs - start) / total_duration) * 100);
		if percentage >= last_percentage + 5:
			print(str(percentage) + " %")
			last_percentage = percentage

	for b in opened_bags:
		b.close()
	bag.close();


if __name__ == "__main__":
	path = sys.argv[1] #where is bag
	duration = int(sys.argv[2]) #how long shoud one bag be (in seconds)
	splitBag(path, duration)