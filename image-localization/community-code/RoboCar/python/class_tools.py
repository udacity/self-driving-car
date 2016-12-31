class Point(object):
	filename=' '
	lat=0
	lon=0
	def init(self, sfilename=None, slat=None, slon=None):
		self.filename = sfilename
		self.lat = slat
		self.lon = slon
	def set_name(self, flnm):
		self.filename=flnm
	def set_lat(self, slat):
		self.lat=slat
	def set_lon(self, slon):
		self.lon=slon
	def length(self):
		return len(self)
	def print_point(self):
		print self.lat, self.lon
	#TODO add pointer number

class Pointer(object):
	mean_lat=0
	mean_lon=0
	n_point=0
	coord=[]
	size_lat=0
	size_lon=0
	def __init__(self):
		self.coord=[]
	def update_pointer(self):
		slat=0
		slon=0
		lat_list=[]
		lon_list=[]
		lat_max=0
		lat_min=0
		lon_max=0
		lon_min=0
		for i in range(0, self.n_point):
			slat+=self.coord[i].lat
			lat_list.append(self.coord[i].lat)
			slon+=self.coord[i].lon
			lon_list.append(self.coord[i].lon)
		self.mean_lat=slat/self.n_point
		self.mean_lon=slon/self.n_point
		self.size_lat=abs(max(lat_list)-min(lat_list))/2
		self.size_lon=abs(max(lon_list)-min(lon_list))/2
	def add_point(self, point):
		self.coord.append(point)
		self.n_point+=1
	def print_pointer(self):
		print "The pointer has ", self.n_point, "points"
		for i in range (0, self.n_point):
			print self.coord[i].lat, self.coord[i].lon, self.coord[i].filename
		print "Mean lat: ", self.mean_lat, "Mean lon: ", self.mean_lon
		print

#Calculate distance between mean lat and long of two pointers
def pointers_dist(pa, pb):
	x1=pa.mean_lat
	x2=pb.mean_lat
	y1=pa.mean_lon
	y2=pb.mean_lon
	return ((x2-x1)**2+(y2-y1)**2)**0.5

#Merge pa to pb pointers
def merge_pointers(pa,pb):
	for i in range(0, pa.n_point):
		pb.add_point(pa.coord[i])
	pb.update_pointer()

def pointers_min_dist(pointers):
	dist_min=100
	pa=0
	pb=0
	dist_min_list=[]
	for i in range(0, len(pointers)):
		for j in range(i, len(pointers)):
			dist=pointers_dist(pointers[i], pointers[j])
			if dist>0:
				if dist<dist_min:
					dist_min=dist
					pa=j
					pb=i

def pointers_min_dist_fast(pointers):
	dist_min=100
	pa=0
	pb=0
	for i in range(0, len(pointers)):
		for j in range(i, len(pointers)):
			dist=pointers_dist(pointers[i], pointers[j])
			if dist>0:
				if dist<merge_dist:
					dist_min=dist
					pa=j
					pb=i
					return pa,pb,dist_min
	return pa,pb,dist_min

#we have more then 3 pointers
def nearest_pointers(pointers, i):
	dist_min1=100
	dist_min2=100
	id1=0
	id2=0
	for j in range(0, len(pointers)):
		dist=pointers_dist(pointers[i], pointers[j])
		if dist>0:
			if dist<dist_min1:
				dist_min1=dist
				id1=j
	for j in range(0, len(pointers)):
		dist=pointers_dist(pointers[i], pointers[j])
		if dist>dist_min1:
			if dist<dist_min2:
				dist_min2=dist
				id2=j
	return id1, dist_min1, id2, dist_min2
