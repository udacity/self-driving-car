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

#include "Validator.h"

namespace Validator
{
namespace detail
{
template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits> &operator<<(
	std::basic_ostream<CharT, Traits> &os,
	const utils::GpsCoords& coords)
{
	return std::cout << "[" << coords.GetLatitude() << ", "
					 		<< coords.GetLongitude() << ", "
					 		<< coords.GetAltitude() << "]";
}
} // detail

using namespace detail;

Validator::Validator(ros::NodeHandle& node) :
node_(node),
bagSub_(node.subscribe(utils::Topics::GPS, 1, &Validator::OnBagGpsCoords, this)),
slamSub_(node_.subscribe(utils::Topics::SlamGPS, 1, &Validator::OnSlamGpsCoords, this)),
pubMSE_(node.advertise<std_msgs::Float64>("/validator/mse",1))
{
	std::cout << std::fixed << std::setprecision(4) << std::left;
}

void Validator::OnBagGpsCoords(const sensor_msgs::NavSatFix& msg)
{
	static const size_t bag_msg_ignore_count = 1;
	static size_t bag_msg_count = 0;
	++bag_msg_count;

	Sample bagSample(msg);
	bagSample.stamp = ros::Time::now(); // time synchronization

	if(!samples_.empty())
	{
		double bagSpeed =
			3.6 * (cv::norm(prevBagSample_.coords.Distance2DTo(bagSample.coords)) /
				(bagSample.stamp - prevBagSample_.stamp).toSec());

		std::cout << "bag: " << prevBagSample_.coords << ", " << bagSample.coords
				  << ", speed: " << bagSpeed << " km/h, time: " << (bagSample.stamp - prevBagSample_.stamp).toSec() << "\n";
		std::cout << std::setw(32) << "bag"
				  << std::setw(32) << "slam"
				  << std::setw(24) << "diff"
				  << "error\n";

		auto bagDist = prevBagSample_.coords.Distance2DTo(bagSample.coords);

		for(auto const& sample : samples_)
		{
			auto bagCoords = prevBagSample_.coords;
			bagCoords.Shift(bagDist * (sample.stamp - prevBagSample_.stamp).toSec());

			auto diff = bagCoords.Distance2DTo(sample.coords);
			double error = cv::norm(diff);
			squareErrorSum_ += error * error;
			++frameCount_;

			std::cout << bagCoords << "\t"
					  << sample.coords << "\t"
					  << diff << "\t"
					  << error << "\n";
		}

		std_msgs::Float64 msg;
		msg.data = std::sqrt(squareErrorSum_ / frameCount_);
		pubMSE_.publish(msg);
		std::cout << "RMS error: " << msg.data << std::endl;

		samples_.clear();
	}

	if(bag_msg_count > bag_msg_ignore_count) {
		prevBagSample_ = bagSample; // perhaps swap
	}
}

void Validator::OnSlamGpsCoords(const sensor_msgs::NavSatFix& msg)
{
	if(!prevBagSample_.coords.isNull() && prevBagSample_.stamp < ros::Time::now())
	{
		samples_.emplace_back(msg);
		samples_.back().stamp = ros::Time::now(); // time synchronization
	}
}
} // Validator
