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
 * @author: RoboAuto team
 * @brief: A node for our motion model's results validation.
*/

#pragma once

#include <Motion.h>

#include <sensor_msgs/Image.h>
#include <geometry_msgs/TwistStamped.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <ros/publisher.h>

#include <vector>

namespace MotionValidator
{
class MotionValidator
{
public:
	MotionValidator(ros::NodeHandle& node);

private:
	void OnCameraUpdate(const sensor_msgs::ImagePtr& msg);

	void OnGpsSpeedUpdate(const geometry_msgs::TwistStampedPtr& msg);

	ros::NodeHandle &node_;
	ros::Subscriber subCamera_;
	ros::Subscriber subGpsSpeed_;

	Motion::Motion motion_;

	bool hasReceivedGpsSpeed_ = false;
	double curGpsSpeed_	= 0;

	double motionDistance_			= 0;
	double gpsDistance_				= 0;
	double maxDistanceError_		= 0;
	double distanceSquareErrorSum_	= 0;

	double errorSum_		= 0;
	double squareErrorSum_	= 0;
	size_t frameCount_		= 0;
};
} // MotionValidator