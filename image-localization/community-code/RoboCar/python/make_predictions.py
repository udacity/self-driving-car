import os
import cv2
import glob
import time
import lmdb
import caffe
import numpy as np
from img_tools import *
from caffe.proto import caffe_pb2
import csv
from operator import add

caffe.set_mode_gpu() 

N_POINTERS=10 #Number of most probable predicted poinerts used to find out actual pointer of the image
MAX_DIST=120

direct=1

#Frames from the test dataset to process
#You can use this parameter to speed up testing process by predictiong only on the first NUMBER_OF_FRAMES images
NUMBER_OF_FRAMES = 6601

#Load pointers
pointers_lat=[]
pointers_lon=[]
ids1=[]
ids2=[]
nframes=[]
with open('./pointers/pointers_SFMV.csv', 'rb') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',')
		for row in reader:
			pointers_lat.append(float(row['mean lat']))
			pointers_lon.append(float(row['mean lon']))
			ids1.append(int(row['id1']))
			ids2.append(int(row['id2']))
			nframes.append(int(row['# of frames']))

pointers_lat_2=[]
pointers_lon_2=[]
ids1_2=[]
ids2_2=[]
nframes_2=[]
with open('./pointers/pointers_MVSF.csv', 'rb') as csvfile:
		reader = csv.DictReader(csvfile, delimiter=',')
		for row in reader:
			pointers_lat_2.append(float(row['mean lat']))
			pointers_lon_2.append(float(row['mean lon']))
			ids1_2.append(int(row['id1']))
			ids2_2.append(int(row['id2']))
			nframes_2.append(int(row['# of frames']))


def label2coord(p_id):
	if direct==1:
		return [pointers_lat[p_id], pointers_lon[p_id]]
	else:
		return [pointers_lat_2[p_id], pointers_lon_2[p_id]]

#Calculate approximate distance between two points
def dist(p1,p2):
	return 100000*((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5


#Rerurn 1 if point with coordinates coord1->coord2 is on the way SF->MV, 2 - SF<-MV
def direction(coord1, coord2):
	if ((coord2[0]-coord1[0])<=0) and ((coord2[1]-coord1[1])>=0):
		return 1
	elif ((coord2[0]-coord1[0])>=0) and ((coord2[1]-coord1[1])<=0):
		return 2
	else:
		return 0

#return coordinates of the nearest pointer to the p_id
def nearest(p_id):
	if direct==1:
		pointer=label2coord(p_id)
		pointer1=label2coord(ids1[p_id])
		if direction(pointer, pointer1)==direct:
			pointer_id=ids1[p_id]
		else:
			pointer_id=ids2[p_id]
		return pointer_id
	elif direct==2:
		pointer=label2coord(p_id)
		pointer1=label2coord(ids1_2[p_id])
		if direction(pointer, pointer1)==direct:
			pointer_id=ids1_2[p_id]
		else:
			pointer_id=ids2_2[p_id]
		return pointer_id
	else:
		return 0

def nearest2(p_id):
	if direct==1:
		pointer=label2coord(p_id)
		pointer1=label2coord(ids1[p_id])
		if direction(pointer, pointer1)==direct:
			pointer_id=ids2[p_id]
		else:
			pointer_id=ids1[p_id]
		return pointer_id
	elif direct==2:
		pointer=label2coord(p_id)
		pointer1=label2coord(ids1_2[p_id])
		if direction(pointer, pointer1)==direct:
			pointer_id=ids2_2[p_id]
		else:
			pointer_id=ids1_2[p_id]
		return pointer_id
	else:
		return 0

#calculate averange shift between frames
def delta2_coord(p1,p2_id):
	p2=label2coord(p2_id)
	p3=label2coord(nearest(p2_id))
	
	coords_tmp=map(add, p2, p3)
	p2p3=[x / 2 for x in coords_tmp]
	n=49
	if direct==1:
		return -abs(p1[0]-p2p3[0])/n, abs(p1[1]-p2p3[1])/n
	else:
		return abs(p1[0]-p2p3[0])/n, -abs(p1[1]-p2p3[1])/n




dxdy=[]
for i in range(1, len(pointers_lat)-1):
	direct=1
	p1=label2coord(i-1)
	p2=label2coord(i)
	p3=label2coord(i+1)
	a1=[abs(p1[0]+p2[0])/2,abs(p1[1]+p2[1])/2]
	a2=[abs(p2[0]+p3[0])/2,abs(p2[1]+p3[1])/2]
	n=49
	dxdy.append([-abs(a1[0]-a2[0])/n, abs(a1[1]-a2[1])/n])
	
dxdy_2=[]
for i in range(1, len(pointers_lat_2)-1):
	direct=2
	p1=label2coord(i-1)
	p2=label2coord(i)
	p3=label2coord(i+1)
	a1=[abs(p1[0]+p2[0])/2,abs(p1[1]+p2[1])/2]
	a2=[abs(p2[0]+p3[0])/2,abs(p2[1]+p3[1])/2]
	n=49
	dxdy_2.append([abs(a1[0]-a2[0])/n, -abs(a1[1]-a2[1])/n])


#Read mean images, caffe model architectures and their trained weights 
mean_blob = caffe_pb2.BlobProto()
with open('./input/SFMV/mean.binaryproto') as f:
	mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
	(mean_blob.channels, mean_blob.height, mean_blob.width))
net = caffe.Net('./caffe/SFMV/deploy.prototxt',
				'./caffe/SFMV/googlenet_train_iter_186000.caffemodel',
				caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

#Read mean images, caffe model architectures and their trained weights 
mean_blob_way = caffe_pb2.BlobProto()
with open('./input/way/mean.binaryproto') as k:
	mean_blob_way.ParseFromString(k.read())
mean_array_way = np.asarray(mean_blob_way.data, dtype=np.float32).reshape(
	(mean_blob_way.channels, mean_blob_way.height, mean_blob_way.width))
net_way = caffe.Net('./caffe/way/deploy.prototxt',
				'./caffe/way/googlenet_train_iter_60000.caffemodel',
				caffe.TEST)

#Define image transformers
transformer_way = caffe.io.Transformer({'data': net_way.blobs['data'].data.shape})
transformer_way.set_mean('data', mean_array_way)
transformer_way.set_transpose('data', (2,0,1))


#Read mean images, caffe model architectures and their trained weights 
mean_blob_2 = caffe_pb2.BlobProto()
with open('./input/MVSF/mean.binaryproto') as h:
	mean_blob_2.ParseFromString(h.read())
mean_array_2 = np.asarray(mean_blob_2.data, dtype=np.float32).reshape(
	(mean_blob_2.channels, mean_blob_2.height, mean_blob_2.width))
net_2 = caffe.Net('./caffe/MVSF/deploy.prototxt',
				'./caffe/MVSF/googlenet_train_iter_186000.caffemodel',
				caffe.TEST)

#Define image transformers
transformer_2 = caffe.io.Transformer({'data': net_2.blobs['data'].data.shape})
transformer_2.set_mean('data', mean_array_2)
transformer_2.set_transpose('data', (2,0,1))


#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("./input/Test/center/*")]
test_img_paths.sort()





#Making predictions
test_ids = []
preds = []
top_k = []
i=0



with open('./submission.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(["frame_id","usrlat","usrlon"])
	counter=0
	path=[]
	npath=[]
	#initially we have one image in the first pointer
	npath_counter=1
	#Start timer
	start_time = time.time()
	#The main loop
	for img_path in test_img_paths:
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		img = transform_img(img)
		#If this is the first image, we have to init the algorithm
		if i==0:
			#Determine our direction
			net_way.blobs['data'].data[...] = transformer_way.preprocess('data', img)
			out_way = net_way.forward()
			direct_tmp=list(net_way.blobs['prob'].data[0].flatten().argsort())[1]
			if direct_tmp==1:
				direct=1 #SF->MV
			else:
				direct=2 #MV->SF
			if direct==1:
				net.blobs['data'].data[...] = transformer.preprocess('data', img)
				out = net.forward()
				top_k.append(list(net.blobs['prob'].data[0].flatten().argsort()[-1:-(N_POINTERS+1):-1]))
			elif direct==2:
				net_2.blobs['data'].data[...] = transformer_2.preprocess('data', img)
				out = net_2.forward()
				top_k.append(list(net_2.blobs['prob'].data[0].flatten().argsort()[-1:-(N_POINTERS+1):-1]))
			test_ids = test_ids + [img_path.split('/')[-1]]
			cur=top_k[0][0]
			path.append(cur)
			frame_coords=label2coord(cur)
			coords_tmp=map(add, label2coord(cur), label2coord(nearest2(cur)))
			frame_coords=[x / 2 for x in coords_tmp]
			d=delta2_coord(frame_coords, path[-1])
		else:
			if direct==1:
				net.blobs['data'].data[...] = transformer.preprocess('data', img)
				out = net.forward()
				top_k.append(list(net.blobs['prob'].data[0].flatten().argsort()[-1:-(N_POINTERS+1):-1]))
			else:
				net_2.blobs['data'].data[...] = transformer_2.preprocess('data', img)
				out = net_2.forward()
				top_k.append(list(net_2.blobs['prob'].data[0].flatten().argsort()[-1:-(N_POINTERS+1):-1]))
			test_ids = test_ids + [img_path.split('/')[-1]]
			lmin=100000 #Just a big number
			jmin=0
			npath_counter+=1
			updated=False
			cur_coord=label2coord(cur)
			for j in range(0,N_POINTERS): 
				l_coord=label2coord(top_k[i][j])
				ldist=dist(cur_coord, l_coord)
				if ldist<lmin and (direction(cur_coord, l_coord)==direct):
					lmin=ldist
					jmin=j
					updated=True
			if updated and lmin<MAX_DIST:
				nex=top_k[i][jmin]
			else:
				nex=cur
			if nex != path[-1]:
				path.append(nex)
				npath.append(npath_counter)
				npath_counter=0
				coords_tmp=map(add, label2coord(path[-1]), label2coord(path[-2]))
				frame_coords=[x / 2 for x in coords_tmp]
				if direct==1:
					d=dxdy[path[-1]-1]
				else:
					d=dxdy_2[path[-1]-1]
				cur=nex
			else:
				#If there are more frames in the pointer than it was in the training ride, then stop at the end of the pointer and don't move forward.
				if npath_counter>50:
					d=[0,0]
				frame_coords=map(add, frame_coords, d)
			writer.writerow([test_ids[counter].split(".png",1)[0], frame_coords[0], frame_coords[1]])
			counter+=1
		i+=1
		if i>NUMBER_OF_FRAMES:
			break
	npath.append(npath_counter)
	frame_coords=map(add, frame_coords, d)
	writer.writerow([test_ids[counter].split(".png",1)[0], frame_coords[0], frame_coords[1]])


timer=time.time() - start_time
print("It was %s seconds long" % timer)
print("Average rate was %s fps" % (NUMBER_OF_FRAMES/timer))
print("%s images were processed" % NUMBER_OF_FRAMES) 
