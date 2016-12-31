import csv
import os
import argparse
import cPickle as pickle


FILE_TO_OPEN = 'interpolated.csv'
FILE_TO_WRITE='pointers2.csv'
FILE_DATA='data.pkl'
NUMBER_OF_PICS = 50
merge_dist=8e-5

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

def main():
	parser = argparse.ArgumentParser(description='Create pointers')
	parser.add_argument('-o', '--outfile', type=str, nargs='?', default='/pointers.csv', help='Output file')
	parser.add_argument('-od', '--outdata', type=str, nargs='?', default='/data.pkl', help='Output data file about pointers')
	parser.add_argument('-i', '--infile', type=str, nargs='?', default='/interpolated.csv', help='Input file')
	arg = parser.parse_args()
	FILE_TO_OPEN=arg.infile
	FILE_TO_WRITE=arg.outfile
	FILE_DATA=arg.outdata
	
	#Read the input file
	point_list=[]
	with open(FILE_TO_OPEN, 'rb') as csvfile:
		interp_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
		for row in interp_reader:
			filename = row['filename']
			#Use only central camera
			if 'left' in filename or 'right' in filename:
				continue
			lat = float(row['lat'])
			lon = float(row['long'])
			if (lat != 0) and (lon != 0):
				p=Point()
				p.init(filename, lat, lon)
				point_list.append(p)
	print "We have ", len(point_list), "points"
	#Create pointers list
	pointer_list=[]
	pointer_list.append(Pointer())
	npointer=0
	for i in range(0, len(point_list)):
		p=point_list[i]
		pointer_list[npointer].add_point(p)
		if (i+1)%NUMBER_OF_PICS==0:
			npointer+=1
			pointer_list.append(Pointer())
	for i in range(0, len(pointer_list)):
		pointer_list[i].update_pointer()
		#pointer_list[i].print_pointer()
	print "We have ", len(pointer_list), "pointers initially"
	pa, pb, dist_min=pointers_min_dist_fast(pointer_list)
	print pa, pb, dist_min
	while dist_min<merge_dist:
		merge_pointers(pointer_list[pa],pointer_list[pb])
		print "Merged", pa, pb
		pointer_list.pop(pa)
		pa, pb, dist_min=pointers_min_dist_fast(pointer_list)
		print pa, pb, dist_min
	print "We have ", len(pointer_list), "pointers at the end"
	print "Writing file"
	nearest1=[]
	nearest2=[]
	sizes_lat=[]
	sizes_lon=[]
	frames=[]
	with open(FILE_TO_WRITE, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(["i", "mean lat", "mean lon", "size lat", "size lon", "id1", "dist1", "id2", "dist2", "# of frames"])
		i=0;
		for p in pointer_list:
			d=nearest_pointers(pointer_list, i)
			nearest1.append(d[1])
			nearest2.append(d[3])
			sizes_lat.append(p.size_lat)
			sizes_lon.append(p.size_lon)
			frames.append(p.n_point)
			writer.writerow([i, p.mean_lat, p.mean_lon, p.size_lat, p.size_lon, d[0], d[1], d[2], d[3], p.n_point])
			i+=1
		print(["param", "min", "aver", "max"])
		print(["size lat", min(sizes_lat)*100000, sum(sizes_lat)*100000/float(len(sizes_lat)), max(sizes_lat)*100000])
		print(["size lon", min(sizes_lon)*100000, sum(sizes_lon)*100000/float(len(sizes_lon)), max(sizes_lon)*100000])
		print(["nearest dist1", min(nearest1)*100000, sum(nearest1)*100000/float(len(nearest1)), max(nearest1)*100000])
		print(["nearest dist2", min(nearest2)*100000, sum(nearest2)*100000/float(len(nearest2)), max(nearest2)*100000])
		print(["frames", min(frames), sum(frames)/float(len(frames)), max(frames)])
	print "File written"
	##Save pointers position
	with open(FILE_DATA, 'wb') as output:
		for p in pointer_list:
			pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)






if __name__ == '__main__':
   main()
