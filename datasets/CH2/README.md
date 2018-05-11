# CH2_002
## ROSBAG training data with very similar driving conditions to CH2_001
| Date | Lighting Conditions | Duration | Compressed Size | Direct Download | Torrent | MD5 |
| ---- | :------------------:| --------:| ---------------:|:---------------:|:-------:|:---:|
| 11/18/2016 | Daytime/Shadows | -- | 4.4GB | None | [Torrent](https://github.com/udacity/self-driving-car/blob/master/datasets/CH2/Ch2_002.tar.gz.torrent) | f3178f88d9b970ed19be01d700b52a9f |

* HMB_1: 221 seconds, direct sunlight, many lighting changes. Good turns in beginning, discontinuous shoulder lines, ends in lane merge, divided highway
* HMB_2: 791 seconds, two lane road, shadows are prevalent, traffic signal (green), very tight turns where center camera can't see much of the road, direct sunlight, fast elevation changes leading to steep gains/losses over summit. Turns into divided highway around 350s, quickly returns to 2 lanes
* HMB_4: 99 seconds, divided highway segment of return trip over the summit
* HMB_5: 212 seconds, guardrail and two lane road, shadows in beginning may make training difficult, mostly normalizes towards the end
* HMB_6: 371 seconds, divided multi-lane highway with a fair amount of traffic

# CH2_001
## Final Round Test Data: JPG and Filtered ROSBAG
| Date | Lighting Conditions | Duration | Compressed Size | Direct Download | Torrent | MD5 |
| ---- | :------------------:| --------:| ---------------:|:---------------:|:-------:|:---:|
| 11/18/2016 | Daytime/Shadows | 280s | 456 MB | Avail. Soon | [Torrent](https://github.com/udacity/self-driving-car/blob/master/datasets/CH2/Ch2_001.tar.gz.torrent) | 844ad71fe6a44eb50d1183aed0e71efc |

The '/center' folder contains JPG images for each test frame, similar to the Round 1 evaluation set. The decision was made to move to JPG from PNG to save file space, and should not cause any artifacting due to the size of the images. If you would like to ge tthe image set in PNG, you can either convert the folder in batch using a tool like IRFANVIEW, or use rwightman's Udacity Reader docker tool (https://github.com/rwightman/udacity-driving-reader) using the PNG flag. I suggest using a batch convert process to avoid complications with not having the original BAG file with all topics, as you will likely need to modify the code in the docker tool. 

The file 'final_example.csv' includes a template CSV file with false values for the steering angle column, but the frame IDs are correct. You should submit you results EXACTLY as provided, as I will not be fixing submissions for the final round. When you have output from your model, please send your results to self-driving-car@udacity.com with the csv attached, named as 'teamName.csv' and your team members in the body of the email.

The leaderboard can be found at the [same location as before](https://github.com/udacity/self-driving-car/tree/master/challenges/challenge_2) and every attempt will be made to update as often as possible. If someone in the community wants to build an automated submission system, please contact me and I'd be happy to work with you on this. You'll get massive props on the Udacity github.

HMB_3.bag is not included, as it is the test dataset, but I've included a filtered HMB_3_release.bag which only includes the center camera imagery, camera info topic, and timing information. I know many of you asked for this last time.

The HMB_3_release.bag file was created using the following filter rules: 

```
rosbag filter HMB_3.bag HMB_3_release.bag "topic == '/center_camera/camera_info' or topic == '/center_camera/image_color/compressed'"
```

