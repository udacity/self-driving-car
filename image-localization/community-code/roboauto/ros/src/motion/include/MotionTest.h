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
 * @brief: A helper class used for evaluation and tweking of the output values of the motion prediction NN.
 */

#pragma once

#include <iostream>
#include <fstream>

#include <utils/Camera.h>

#include <utils/Topics.h>
#include <utils/OpticalFlow.h>
#include <utils/GpsCoords.h>

#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TwistStamped.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>

#include <NeuralNetwork/FeedForward/Network.h>
#include <NeuralNetwork/Learning/BackPropagation.h>
#include <NeuralNetwork/ActivationFunction/HyperbolicTangent.h>

#include "Sample.h"
#include "Config.h"
#include "Motion.h"


namespace Motion {
    class MotionTest {
    public:
        MotionTest(int argc, char** argv, ros::NodeHandle &n) : n_(n), net_(0){
            subCamImage_ = n_.subscribe(utils::Topics::CAMERA, 1, &MotionTest::OnCameraImage, this);
            gpsSub_ = n_.subscribe(utils::Topics::GPS, 1, &MotionTest::OnGpsUpdate, this);
            gpsSpeedSub_ = n_.subscribe(utils::Topics::GPS_SPEED, 1, &MotionTest::OnGpsSpeedUpdate, this);

            for(int i=0; i<argc; i++){
                if(argv[i][1] == 'l'){
                    loadFile = std::string(argv[++i]);
                }else {
                    std::cout << "No file with the NN specified!\n";
                }
            }

            std::cout << "Load file: " << loadFile << std::endl;
            std::cout << "MotionTest ready..." << std::endl;
        };

        float GetDistEstimation(){ return -1; };
    protected:
        void OnCameraImage(const sensor_msgs::ImageConstPtr& msg){

            if((frameCnt++)%2 == 0)
                return;

            cv::Mat orig;
            utils::Camera::ConvertImage(msg, orig);

            motion.OnImage(orig);

            motion.GetDist();
        };

        void OnGpsUpdate(const sensor_msgs::NavSatFixPtr& msg){
            lat_.push_back(msg->latitude);
            lon_.push_back(msg->longitude);

            utils::GpsCoords coord(msg->latitude, msg->longitude, 0);

            coords_.push_back(coord);
            gpsTime_.push_back(msg->header.stamp.toSec());
        };
        void OnGpsSpeedUpdate(const geometry_msgs::TwistStampedPtr& msg){
//            std::cout << "Speed\n";
            double vx = msg->twist.linear.x;
            double vy = msg->twist.linear.y;
            //std::cout << "speed: " << sqrt(vx*vx + vy*vy) << std::endl;

            double speed = sqrt(vx*vx + vy*vy);

            speeds_.push_back(speed);
        };

        float speed_ = 0;
        std::vector<cv::Mat> images_;
        ros::NodeHandle &n_;
        ros::Subscriber subCamImage_;
        ros::Subscriber gpsSub_ ;
        ros::Subscriber gpsSpeedSub_ ;

        std::vector<double> lat_;
        std::vector<double> lon_;
        std::vector<double> speeds_;
        std::vector<double> gpsTime_;
        std::vector<utils::GpsCoords> coords_;


        std::string loadFile;
        int frameCnt = 0;
        NeuralNetwork::FeedForward::Network net_;

        Motion motion;

        void initNN(NeuralNetwork::FeedForward::Network& n, const std::string& file);
        void loadNetFromFile(const std::string& file);
        double predictSpeedByNN(const cv::Mat flow);
    };
}
