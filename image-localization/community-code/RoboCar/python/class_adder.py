import csv
import os
import argparse
import cPickle as pickle
from class_tools import *
import shutil
FILE_TO_OPEN = 'interpolated.csv'
FILE_DATA='data.pkl'
FILE_TO_WRITE='pointers_1_smp.csv'


#Maximal number images per pointer 
MAX_POINT=100
#How should we reduce size of pointers
SCALE_POINTER=0.5


#Copy file and rename
def rcopy(old_file_name, new_file_name, old_folder, new_folder):
	shutil.copy(os.path.join(old_folder, old_file_name),new_folder)
	os.rename(os.path.join(new_folder, old_file_name), os.path.join(new_folder, new_file_name))


def main():
	parser = argparse.ArgumentParser(description='Add images to pointers')
	parser.add_argument('-o', '--outfile', type=str, nargs='?', default='/pointers.csv', help='Output file')
	parser.add_argument('-id', '--indata', type=str, nargs='?', default='/data.pkl', help='Input data file about pointers')
	parser.add_argument('-i', '--infile', type=str, nargs='?', default='/interpolated.csv', help='Input file')
	parser.add_argument('-sf', '--source_folder', type=str, nargs='?', default='./', help='Source folder with images in ./center/ folder')
	parser.add_argument('-df', '--destination_folder', type=str, nargs='?', default='./', help='Destination folder where we save files')
	parser.add_argument('-l', '--label', type=str, nargs='?', default='nnn', help='String to add to mark this dataset')
	parser.add_argument('-n', '--n_pointers', type=int, nargs='?', default=1000, help='Number of pointers on the road')
	parser.add_argument('-m', '--max_points', type=int, nargs='?', default=100, help='Maximal number of points per pointer')
	arg = parser.parse_args()
	FILE_TO_OPEN=arg.infile
	FILE_TO_WRITE=arg.outfile
	FILE_DATA=arg.indata
	DST_FOLDER=arg.destination_folder
	SRC_FOLDER=arg.source_folder
	LABEL=arg.label
	num_of_pointers=arg.n_pointers
	MAX_POINT=arg.max_points
	#Read the input file
	point_list=[]
	pointer_list=[]

	with open(FILE_DATA, 'rb') as input:
		for i in range (0, num_of_pointers):
			pointer_list.append(pickle.load(input))
			pointer_list[i].n_point=0
			pointer_list[i].coord=[]

	point_list2=[]
	with open(FILE_TO_OPEN, 'rb') as csvfile:
		interp_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
		for row in interp_reader:
			filename = row['filename']
			#Use only central camera
			if 'left' in filename or 'right' in filename:
				continue
			ss=filename.split("center/",1)[1]
			s=int(ss.split(".",1)[0])
			#Uncomment for the second dataset (Ch2-Train)
			#if s>1477435118305680000: #'<'for SF->MV path, use '>' for MV->SF
				#continue
			lat = float(row['lat'])
			lon = float(row['long'])
			if (lat != 0) and (lon != 0):
				p=Point()
				p.init(ss, lat, lon)
				point_list2.append(p)
	print "We have ", len(point_list2), "points in the dataset"
	
	for p in point_list2:
		x=p.lat
		y=p.lon
		for pp in pointer_list:
			xmax=pp.mean_lat+SCALE_POINTER*pp.size_lat
			xmin=pp.mean_lat-SCALE_POINTER*pp.size_lat
			ymax=pp.mean_lon+SCALE_POINTER*pp.size_lon
			ymin=pp.mean_lon-SCALE_POINTER*pp.size_lon
			if (x<=xmax) and (x>=xmin) and (y<=ymax) and (y>=ymin) and (pp.n_point<MAX_POINT):
				 pp.add_point(p)
				 break
	
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
	
	n_files=0
	#Rename and copy files
	print "Copy files..."
	for i in range(0, len(pointer_list)):
		if len(pointer_list[i].coord)>0:
			for j, p in enumerate(pointer_list[i].coord):
				#print p.filename
				rcopy(p.filename,(str(i)+'_'+str(j)+LABEL+'.jpg'),SRC_FOLDER,DST_FOLDER)
				n_files+=1
	print "Complete!"
	print n_files, "frames added"





if __name__ == '__main__':
   main()
