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
 * @brief: A GPX exporter module.
 */

#include <utils/Topics.h>
#include <sensor_msgs/NavSatFix.h>
#include <ros/ros.h>
#include <chrono>

#include <time.h>

ros::Subscriber subGPS_;

void OnGPS(const sensor_msgs::NavSatFix& msg)
{
    time_t now = msg.header.stamp.sec;
    std::cout << "<trkpt lat=\"" << std::setprecision(10) << msg.latitude << "\" lon=\"" << std::setprecision(10) << msg.longitude <<"\"><ele> " << msg.altitude << "</ele>"
            "<time>" << std::put_time(std::localtime(&now), "%Y-%m-%dT%H:%M:%SZ") << "</time></trkpt>\n";
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gpx");
    ros::NodeHandle node ("gpx");

    std::cout << std::setprecision(10);

    std::cout << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<gpx version=\"1.0\">\n"
            "\t<name>Example gpx</name>"
            "<trk><name>Example gpx</name><number>1</number><trkseg>\n";
    subGPS_ = node.subscribe(utils::Topics::GPS, 1, OnGPS);
    ros::spin();
    std::cout << "\t</trkseg></trk>\n"
            "</gpx>";
}