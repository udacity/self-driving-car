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
 * @brief: A helper class used for training the NNs for motion prediction.
 */

#pragma once

#include <iostream>
#include <fstream>

#include <utils/Camera.h>

#include <utils/Topics.h>
#include <utils/OpticalFlow.h>

#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>

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

namespace Motion {
    class MotionLearner{
    public:

        MotionLearner(int argc, char** argv, ros::NodeHandle &n) : net_(INPUT_SIZE), n_(n)
        {
            cameraSub_ = n_.subscribe(utils::Topics::CAMERA, 1, &MotionLearner::OnCameraUpdateDataset, this);
            gpsSub_ = n_.subscribe(utils::Topics::GPS, 1, &MotionLearner::OnGpsUpdate, this);
            gpsSpeedSub_ = n_.subscribe(utils::Topics::GPS_SPEED, 1, &MotionLearner::OnGpsSpeedUpdate, this);
            commandSub_ = n_.subscribe("/motion/nn", 1, &MotionLearner::CommandReceived, this);
            // a speed is 0 until first speed update?
            speeds_.push_back(0);

            for(int i=0; i<argc; i++){
                if(argv[i][1] == 'l'){
                    loadFile = std::string(argv[++i]);
                }else if(argv[i][1] == 's'){
                    saveFile = std::string(argv[++i]);
                }
            }

            std::cout << "Save file: " << saveFile << std::endl;
            std::cout << "Load file: " << loadFile << std::endl;

            initNN(net_, loadFile);

            std::cout << "MotionLearner ready...\n";
        };
    private:

        int nn_learning_iterations_ = 1000;

        const int INPUT_SIZE = 336*2;

        bool learning;
        int frameCnt = 0;

        void OnCameraUpdateDataset(const sensor_msgs::ImageConstPtr& msg);
        void OnCameraUpdatePredict(const sensor_msgs::ImageConstPtr& msg);
        void OnGpsUpdate(const sensor_msgs::NavSatFixPtr& msg);
        void OnGpsSpeedUpdate(const geometry_msgs::TwistStampedPtr& msg);
        void CommandReceived(const std_msgs::StringPtr& msg);

        void addSample(cv::Mat flow, double speed);
        void initNN(NeuralNetwork::FeedForward::Network& n, const std::string& file);
        void learn(NeuralNetwork::FeedForward::Network& n);
        void validate(NeuralNetwork::FeedForward::Network &n, std::vector<Sample::Sample> &valid);
        void saveNetToFile(const std::string& file);
        void loadNetFromFile(const std::string& file);
        double predictDistance(cv::Mat flow);

        NeuralNetwork::FeedForward::Network net_;

        ros::NodeHandle &n_;
        ros::Subscriber cameraSub_;
        ros::Subscriber gpsSub_;
        ros::Subscriber gpsSpeedSub_;
        ros::Subscriber commandSub_;

        std::vector<cv::Mat> images_;
        std::vector<double> imgStamps_;
        std::vector<cv::Mat> flows_;

        std::vector<double> speeds_;
        std::vector<double> speedStamps_;

        std::vector<Sample::Sample> samples_;

        std::string loadFile;
        std::string saveFile;
    };
}
