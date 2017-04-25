# Dataset Wiki
In an attempt to cleanup the data release practices of the Udacity Self-Driving Car team, we will start maintaining this wiki of data we have uploaded. Issues persist throughout many of the datasets, so we will be working backwards to catalog legacy data and unify the naming methodology. Please feel free to add to this list and submit a PR.

Check out [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader) for some easy-to-use scripts to read or export to CSV or TensorFlow.

## Current Releases
These releases should be issue/error free and comply with the new naming schema.

#### Challenge 2 Driving Data
| Name | Purpose |
|:----:|:-------:|
| [CH2_001](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2) | Final testing data for the last round of Challenge 2 |
| [CH2_002](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2) | Training data with very similar driving conditions to Ch2_001 |

#### Challenge 3 Driving Data
| Name | Purpose |
|:----:|:-------:|
| [CH3_001](https://github.com/udacity/self-driving-car/tree/master/datasets/CH3) | Northbound and Southbound drives on El Camino with IMU positioning |
| [CH3_002](https://github.com/udacity/self-driving-car/tree/master/datasets/CH3) | Continuous North/South on El Camino with IMU positioning and HDL-32E LIDAR |

#### Misc. Driving Data
| Name | Purpose |
|:----:|:-------:|
| [CHX_001](https://github.com/udacity/self-driving-car/tree/master/datasets/CHX) | Lap around block at Udacity office with new HDL-32E LIDAR |

## Legacy Data

[All torrent releases from Udacity can be found on our AcademicTorrents page with associated descriptions.](http://academictorrents.com/userdetails.php?id=5125) These releases are old and likely have issues that make them unsuitable for training, but many are useful. We will update legacy releases with more info as we move forward, but please use them only as a reference for now if you don't check them beforehand.

#### Driving Data
| Date | Lighting Conditions | Duration | Compressed Size | Uncompressed | Direct Download | Torrent | MD5 |
| ---- | :------------------:| --------:| ---------------:| ------------:|:---------------:|:-------:|:---:|
| 09/29/2016 | Sunny | 00:12:40 | 25G | 40G | [HTTP](http://bit.ly/udacity-dataset-2-1) | [Torrent](datasets/dataset.bag.tar.gz.torrent)| `33a10f7835068eeb29b2a3274c216e7d` |
| 10/03/2016 | Overcast | 00:58:53 | 124G | 183G | [HTTP](http://bit.ly/udacity-dataset-2-2) | [Torrent](datasets/dataset-2-2.bag.tar.gz.torrent) | `34362e7d997476ed972d475b93b876f3` |
| 10/10/2016 | Sunny | 03:20:02 | 21G | 23.3G |  | [Torrent](http://bit.ly/2dZTOcq) | `156fb6975060f60c452a9fa7c4121195` |
| 10/20/2016 | Sunny | 03:30:00 | 30G | 40G |  | [Torrent](http://bit.ly/2epl7Ir ) | `13f107727bed0ee5731647b4e114a545` |

#### Isolated and Trimmed Driving Data
With the help of [Auro Robotics](http://www.auro.ai/), compression, and selective recording, we now have considerably smaller datasets.
