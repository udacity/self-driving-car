#!/bin/bash

## VERSION WITH SKIPPED OVERCAST!!!

# folder with bags
BAGS_FOLDER="/home/robo/bag"
# maximum secs of bag is played
MAX_DURATION=1000
# NN used for learning
#NN_LOAD="./../ros/src/motion/test2.json"
NN_LOAD=""
NN_SAVE="./../ros/src/motion/test100.json"
# NN iters
NN_ITERS=10000



###################################################

echo
echo
echo "Version with skipped Overcast!"
echo
echo
#sleep 2

###################################################

# run module
#if [ $NN_LOAD="" ]; then
#	bash -c "rosrun motion motionExe -s $NN_SAVE" &
#else
#	bash -c "rosrun motion motionExe -s $NN_SAVE -l $NN_LOAD" &
#fi

#echo

cd $BAGS_FOLDER

# create dataset from all bags
for f in *.bag
do
	if [[ $f == *"Overcast"* ]]; then
		continue
	fi

	echo "Processing $f \nmaximum duration: $MAX_DURATION\n\n"
	rosbag play $f -s 0 --duration $MAX_DURATION
done 

echo "\n\nLearning\n\n"

# learn
for i in {1..2}
do
	# learn
	timeout 1 rostopic pub /motion/nn std_msgs/String "data: 'l $NN_ITERS'"

	echo
	# save
	timeout 1 rostopic pub /motion/nn std_msgs/String "data: 's'"

	echo
done

echo
# save
timeout 1 rostopic pub /motion/nn std_msgs/String "data: 's'"

# leave time for saving?
# sleep 2s

# echo "\n\nKilling motion\n\n"

# rosnode kill motion
# wait
