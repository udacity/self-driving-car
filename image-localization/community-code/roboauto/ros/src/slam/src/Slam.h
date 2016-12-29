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
 * @brief: An extension class for SlamImpl (slam class itself) with added callbacks to the ROS topics.
 */

#pragma once

#include "SlamImpl.h"

#include <utils/Camera.h>
#include <utils/Visualization.h>

#include <utils/Topics.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>

namespace Slam {
    class Slam : public SlamImpl {
    public:
        Slam (ros::NodeHandle &node);
    protected:
        bool PUBLISH_VIS = false;
        bool PUBLISH_IMGS = false;

        bool localize_ = false;

        ros::NodeHandle &node_;

        Motion::Motion motion_;
        utils::Visualization path_;

        ros::Subscriber subCamImage_;
        ros::Subscriber subGPS_;
        ros::Subscriber subLocalize_;
        ros::Subscriber subCommand_;

        ros::Publisher pubGPS_;
        ros::Publisher pubHorizont_;
        ros::Publisher pubDetection_;

        std::shared_ptr<sensor_msgs::NavSatFix> gps_;
        std::unique_ptr<utils::Map::Segment> segment_;

        void VisualizeDetectedObjects(const cv::Mat &img);

        using SlamImpl::Localize;
        std::pair<utils::GpsCoords,double> DetectAndLocalize(const cv::Mat &img);
        void Teach(const cv::Mat &img, const ros::Time &stamp);

        void OnCameraImage(const sensor_msgs::ImageConstPtr& msg);
        void OnGPS(const sensor_msgs::NavSatFix& msg);
        void OnLocalize(const std_msgs::Empty& msg);
        void OnCommand(const std_msgs::String& msg);
    };
}