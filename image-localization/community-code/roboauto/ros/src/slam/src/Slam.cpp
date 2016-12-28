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

#include "Slam.h"

#include <future>
#include <thread>

Slam::Slam::Slam(ros::NodeHandle &node) :
        node_(node),
        path_(node, "/slam") {
    motion_.SetError(tree_.get<double>("motionError"));
    PUBLISH_VIS = tree_.get<bool>("enableVisualisation");
    PUBLISH_IMGS = tree_.get<bool>("enableVisualisation");

    subCamImage_ = node.subscribe(utils::Topics::CAMERA, 1, &Slam::OnCameraImage, this);
    subGPS_ = node.subscribe(utils::Topics::GPS, 1, &Slam::OnGPS, this);
    subLocalize_ = node.subscribe("/localize", 1, &Slam::OnLocalize, this);
    subCommand_ = node.subscribe("/command", 1, &Slam::OnCommand, this);

    pubGPS_ = node.advertise<sensor_msgs::NavSatFix>(utils::Topics::SlamGPS, 1);
    if(PUBLISH_IMGS) {
        pubHorizont_ = node.advertise<sensor_msgs::Image>("/horizont", 1);
    }
    if(PUBLISH_VIS) {
        pubDetection_ = node.advertise<sensor_msgs::Image>("/detection", 1);
    }

    motion_.SetError(tree_.get<double>("motionError"));
    std::cout << "\n==== PF =====\n";
    std::cout << "motionError: " <<  motion_.GetError() << "\n";

}

void Slam::Slam::OnCameraImage(const sensor_msgs::ImageConstPtr &msg) {
    cv::Mat img;
    utils::Camera::ConvertImage(msg, img);

    if (localize_) {
        auto coords = DetectAndLocalize(img);
    } else {
        Teach(img, msg->header.stamp);
    }
}

void Slam::Slam::OnGPS(const sensor_msgs::NavSatFix &msg) {
    gps_ = std::make_shared<sensor_msgs::NavSatFix>(msg);
}

void Slam::Slam::OnLocalize(const std_msgs::Empty &msg) {
    localize_ = true;
    Reset();
    PostProcess();
    std::cout << "Localization: ON\n";
}

void Slam::Slam::OnCommand(const std_msgs::String& msg) {
    if(msg.data[0] == 's') {
        SaveMap(std::string({msg.data.begin()+2, msg.data.end()}));
        std::cout << "Saved map to " <<  std::string({msg.data.begin()+2, msg.data.end()}) << std::endl;
    } else if(msg.data[0] == 'l') {
        map_ = utils::Map::Map();
        LoadMap(std::string({msg.data.begin()+2, msg.data.end()}));

        std::cout << "Loaded map from " <<  std::string({msg.data.begin()+2, msg.data.end()}) << std::endl;
        path_.DrawMap(map_);
    }
}

/**
 * This method performs all the actions associated with localization.
 * At first it describes the image and computes motion estimation.
 * After that it computes new position and updates position values and visualizations.
 * @param img
 */
std::pair<utils::GpsCoords,double> Slam::Slam::DetectAndLocalize(const cv::Mat &img) {

    cv::Mat gray;
    cv::cvtColor(img, gray, CV_BGR2GRAY);

    auto motionAsyncRes = std::async(std::launch::async, [this,img]() {
        // Update motion from img
        motion_.OnImage(img);
        state_.distance = motion_.GetDist();
        state_.distanceError = motion_.GetError();
    });

    auto objectDetectionRes = std::async(std::launch::async, [this,img,gray]() {
        DetectObjects(img,gray);
        if (PUBLISH_IMGS) {
            VisualizeDetectedObjects(img);
        }
    });

    motionAsyncRes.get();
    objectDetectionRes.get();

    auto coords = Localize();

    if (PUBLISH_VIS) {
        /* Draw map from time to time*/
        static int counter = 0;
        if (counter++ % 20 == 0) {
            path_.DrawMap();
        }
        path_.DrawParticles(map_, particleFilter_);
        auto clusters = clusterization_.GetAllClusters();
        path_.DrawClusters(map_, clusters);
        path_.DrawPosition(map_, coords.first);
        path_.DrawAugmentedBorders(map_, augmentedFrom_, augmentedTo_);
    }

    /* Publish coordinates*/
    sensor_msgs::NavSatFix gps;
    gps.header.stamp = ros::Time::now();
    gps.latitude = coords.first.GetLatitude();
    gps.longitude = coords.first.GetLongitude();
    gps.position_covariance[0] = coords.second;
    pubGPS_.publish(gps);
    return coords;
}

/**
 * This method is used for creation of a map from a BAG file.
 * @param img
 * @param stamp
 */
void Slam::Slam::Teach(const cv::Mat &img, const ros::Time &stamp) {
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_BGR2GRAY);

    DetectObjects(img,gray);

    if (PUBLISH_IMGS) {
        VisualizeDetectedObjects(img);
    }

    if (gps_) {
        utils::GpsCoords coords(gps_->latitude, gps_->longitude, 0);
        if (segment_) {
            cv::Point3d distance = segment_->GetBeginning().Distance3DTo(coords);
            if (cv::norm(distance) < 0.1) {
                segment_->AddMapPoint(utils::Map::MapPoint(state_.detectedObjects));
                gps_.reset();
                return;
            }

            segment_->SetDir(distance);
            map_.AddSegment(*segment_);

            if (map_.GetSegmentsSize() > 1) {
                std::size_t lastSegmentID = map_.GetSegmentsSize() - 1;
                map_.GetSegment(lastSegmentID - 1).AddNextSegment(std::make_tuple(1.0, lastSegmentID, 0.0));
            }
        }

        segment_ = std::make_unique<utils::Map::Segment>();
        segment_->SetBeginning(coords);
        segment_->AddMapPoint(utils::Map::MapPoint(state_.detectedObjects));

        if (PUBLISH_VIS) {
            path_.DrawMap(map_);
        }

        gps_.reset();
    } else if(segment_){
        segment_->AddMapPoint(utils::Map::MapPoint(state_.detectedObjects));
    }
}

void Slam::Slam::VisualizeDetectedObjects(const cv::Mat &img) {
    // Horizont publishing
    sensor_msgs::ImagePtr msgHorizont = cv_bridge::CvImage(std_msgs::Header(), "mono8",
                                                           descriptor_.ExtractHorizontByColor(img)).toImageMsg();
    msgHorizont->header.stamp = ros::Time::now();
    pubHorizont_.publish(msgHorizont);

    // Publishing of detected objects
    cv::Mat displayDetectedThings(img);
    img.copyTo(displayDetectedThings);

    descriptor_.descToIMG(displayDetectedThings, state_.detectedObjects.horizont);
    trafficLightDetector_.DisplayTraficLights(displayDetectedThings, state_.detectedObjects.lights);
    poleDetector_.DrawPolesToImage(displayDetectedThings, state_.detectedObjects.poles);

    utils::SignDetection::DrawSignToImg(displayDetectedThings, state_.detectedObjects.yellowSigns, "yellow");
    utils::SignDetection::DrawSignToImg(displayDetectedThings, state_.detectedObjects.darkYellowSigns, "dark yellow");
    utils::SignDetection::DrawSignToImg(displayDetectedThings, state_.detectedObjects.darkGreenSigns, "dark green");

    sensor_msgs::ImagePtr detection = cv_bridge::CvImage(std_msgs::Header(), "bgr8",
                                                         displayDetectedThings).toImageMsg();
    detection->header.stamp = ros::Time::now();
    pubDetection_.publish(detection);
}