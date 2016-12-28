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
 * @brief: A class used for creation of a run dataset. The DS is then used for automated validation.
 */

#include "SlamImpl.h"

#include <utils/Topics.h>

#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/String.h>

class DataSetCreator : public Slam::SlamImpl {
public:
    DataSetCreator(ros::NodeHandle &node) {
        subCamImage_ = node.subscribe(utils::Topics::CAMERA, 1, &DataSetCreator::OnCameraImage, this);
        subGPS_ = node.subscribe(utils::Topics::GPS, 1, &DataSetCreator::OnGPS, this);
        subCommand_ = node.subscribe("/command", 1, &DataSetCreator::OnCommand, this);
    }

    void OnCameraImage(const sensor_msgs::ImageConstPtr& msg) {
        cv::Mat img;
        utils::Camera::ConvertImage(msg, img);

        cv::Mat gray;
        cv::cvtColor(img, gray, CV_BGR2GRAY);

        DetectObjects(img,gray);
        // Update motion from img
        motion_.OnImage(img);
        state_.distance = motion_.GetDist();
        state_.distanceError = motion_.GetError();

        batchSamples_.push_back(std::make_tuple(state_, msg->header.stamp.toSec()));
    }

    void OnGPS(const sensor_msgs::NavSatFix& msg) {
        if(gps_) {
            utils::GpsCoords prevCoords(gps_->latitude, gps_->longitude, gps_->altitude);
            utils::GpsCoords actCoords(msg.latitude, msg.longitude, msg.altitude);

            auto diff = prevCoords.Distance2DTo(actCoords) / (msg.header.stamp - gps_->header.stamp).toSec();

            utils::GpsCoords coords;
            double time;
            CurrentState state;

            // interpolate GPS coords
            for (auto const &sample : batchSamples_) {
                coords = prevCoords;

                std::tie(state, time) = sample;

                // GPS update comes every sec...
                coords.Shift(diff * (time - gps_->header.stamp.toSec()));
                runSamples_.push_back(std::make_tuple(state, coords));
            }

            batchSamples_.clear();
        }
        gps_ = std::make_shared<sensor_msgs::NavSatFix>(msg);
    }

    void OnCommand(const std_msgs::String& msg) {
        SaveRun(msg.data);
        std::cout << "Run saved to " << msg.data << std::endl;
    }

protected:
    void SaveRun(const std::string &filename) {
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oa(ofs);

        oa << runSamples_;
    }

    std::shared_ptr<sensor_msgs::NavSatFix> gps_;
    std::vector<std::tuple<CurrentState, utils::GpsCoords>> runSamples_;
    std::vector<std::tuple<CurrentState, double>> batchSamples_;

    Motion::Motion motion_;

    ros::Subscriber subCamImage_;
    ros::Subscriber subGPS_;
    ros::Subscriber subCommand_;

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "slam");
    ros::NodeHandle node("slam");
    DataSetCreator slam(node);

    ros::spin();
}