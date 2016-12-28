/*
 *  Copyright 2016 RoboAuto team, Artin
 *  All rights reserved.
 *
 *  This file is part of RoboAuto HorizonSlam.
 *
 *  RoboAuto HorizonSlam is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RoboAuto HorizonSlam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RoboAuto HorizonSlam.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @author: RoboAuto Team
 * @brief: A node for comparison of the GPS coordinates published by our system and the GPS coordinates in the *.bag file.
 */

#pragma once

#include <utils/Topics.h>
#include <utils/GpsCoords.h>

#include <ros/ros.h>

#include <std_msgs/String.h>
#include <sensor_msgs/NavSatFix.h>

#include <utility>

#include <std_msgs/Float64.h>

namespace Validator
{
class Validator
{
public:
	Validator(ros::NodeHandle &node);

private:
	struct Sample
	{
		Sample(const sensor_msgs::NavSatFix& msg = {}) :
		stamp(msg.header.stamp),
		coords(msg.latitude, msg.longitude, msg.altitude)
		{
		}

		ros::Time stamp;
		utils::GpsCoords coords;
	};

	void OnBagGpsCoords(const sensor_msgs::NavSatFix& coords);

	void OnSlamGpsCoords(const sensor_msgs::NavSatFix& coords);

	ros::NodeHandle &node_;
	ros::Subscriber bagSub_, slamSub_;
	ros::Publisher pubMSE_;

	Sample prevBagSample_;
	std::vector<Sample> samples_;

	double squareErrorSum_	= 0;
	size_t frameCount_		= 0;
};
} // Validator
