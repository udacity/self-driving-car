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

#include "MotionValidator.h"

static const bool DO_RANDOM_FRAME_DUMP = false;

namespace MotionValidator
{
namespace detail
{
double linearSpeed(const geometry_msgs::TwistStamped& msg)
{
	return std::sqrt(std::pow(msg.twist.linear.x, 2) + std::pow(msg.twist.linear.y, 2));
}
} // detail

MotionValidator::MotionValidator(ros::NodeHandle& node) :
node_(node),
subCamera_(node.subscribe(utils::Topics::CAMERA, 1, &MotionValidator::OnCameraUpdate, this)),
subGpsSpeed_(node.subscribe(utils::Topics::GPS_SPEED, 1, &MotionValidator::OnGpsSpeedUpdate, this))
{
	std::cout << std::fixed << std::setprecision(2) << std::right;
}

void MotionValidator::OnCameraUpdate(const sensor_msgs::ImagePtr& msg)
{
	static std::default_random_engine eng(std::random_device{}());
	static std::uniform_int_distribution<int> dist(0, 1);

	if(DO_RANDOM_FRAME_DUMP && dist(eng))
		return;

	static ros::Time prevStamp;

	if(hasReceivedGpsSpeed_ && prevStamp.isValid())
	{
		cv::Mat orig;
		utils::Camera::ConvertImage(msg, orig);
		motion_.OnImage(orig);

		double gpsDist = curGpsSpeed_ * (msg->header.stamp - prevStamp).toSec();
		gpsDistance_ += gpsDist;

		double dist = motion_.GetDist();
		motionDistance_ += dist;

		double curSpeed_ = dist / (msg->header.stamp - prevStamp).toSec();

		double distanceError = motionDistance_ - gpsDistance_;
		distanceSquareErrorSum_ += distanceError * distanceError;

		if(std::abs(distanceError) > std::abs(maxDistanceError_)) {
			maxDistanceError_ = distanceError;
		}

		errorSum_ += dist - gpsDist;
		squareErrorSum_ += std::pow(dist - gpsDist, 2);
		++frameCount_;
		double mse = squareErrorSum_ / frameCount_;

		std::cout << std::setw(16) << "momentary"
				  << std::setw(10) << "speed"
				  << std::setw(12) << "total"
				  << std::setw(16) << "MSE"
				  << std::setw(8) << "RMSE"
				  << std::setw(8) << "AVG"
				  << "\n";
		std::cout << std::setfill(' ');
		std::cout << std::setw(8) << "GPS:"
				  << std::setw(8) << gpsDist << " m"
				  << std::setw(8) << curGpsSpeed_ << " m/s"
				  << std::setw(8) << gpsDistance_ << " m"
				  << std::setw(14) << mse
				  << std::setw(8) << std::sqrt(mse)
				  << std::setw(8) << (errorSum_ / frameCount_)
				  << std::setw(16) << "RMSE total:"
				  << std::setw(8) << std::sqrt(distanceSquareErrorSum_ / frameCount_) << " m\n"
				  << std::setw(8) << "motion:"
				  << std::setw(8) << dist << " m"
				  << std::setw(8) << curSpeed_ << " m/s"
				  << std::setw(8) << motionDistance_ << " m"
				  << std::setw(46) << "max total:"
				  << std::setw(8) << maxDistanceError_ << " m\n\n";
	}

	prevStamp = msg->header.stamp;
}

void MotionValidator::OnGpsSpeedUpdate(const geometry_msgs::TwistStampedPtr& msg)
{
	std::cout << std::setfill('-');
	hasReceivedGpsSpeed_ = true;
	curGpsSpeed_ = detail::linearSpeed(*msg);
}
}