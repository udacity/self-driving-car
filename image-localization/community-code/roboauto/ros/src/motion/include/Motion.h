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
 * @brief: A class for estiamation of a car motion based on consecutive images.
 * Distance between the frames is calculated by Neural Network.
 * The input of the neural network are subsampled values of dense optical flow, output is predicted distance.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>

#include <utils/Camera.h>

#include <utils/Topics.h>
#include <utils/OpticalFlow.h>

#include <ros/ros.h>

#include <NeuralNetwork/FeedForward/Network.h>
#include <NeuralNetwork/Learning/BackPropagation.h>
#include <NeuralNetwork/ActivationFunction/HyperbolicTangent.h>

#include "Sample.h"
#include "Config.h"

#include <ros/package.h>

namespace Motion {

    class Motion {
    public:
        Motion( const std::string& path = ros::package::getPath("motion") + std::string("/net.json")) : loadFile(path), net_(0) {
            initNN(net_, loadFile);
        };

        float GetError(){ return error_; };

        void OnImage(const cv::Mat &orig) {
            cv::Mat mat;
            resize(orig, mat, cv::Size(orig.cols*SCALE_X, orig.rows*SCALE_Y));

            cv::Mat gray;
            if(mat.type() != 0) {
                cvtColor(mat, gray, CV_BGR2GRAY);
            } else {
                gray=mat;
            }

            images_.push_back(gray);

            if(images_.size() < 2) {
                return;
            } else if (images_.size() > 100){
                std::vector<cv::Mat>(images_.begin()+50, images_.end()).swap(images_);
            }

            flow_ = utils::Motion::getOptFlow(images_[images_.size()-2], images_[images_.size()-1], SAMPLE_STEP_X, SAMPLE_STEP_Y);
            distances_.push_back(predictDistByNN(flow_));
            dist_ = GetDistEstimation();
        }

        float GetDist() const {
            return dist_;
        }

        void SetError(float error ) {
            error_ = error;
        }

        cv::Mat flow_;
    protected:
        float error_ = 0.8;
        float GetDistEstimation(){
            if(distances_.empty())
                return 0;

            if(distances_.size() == 1)
                return distances_[0];

            float dist = 0;
            int cnt = 0;

            for(int i=distances_.size()-1; cnt < winSize_ && i >= 0; i--, cnt++){
                dist += distances_[i];
            }

            dist /= cnt;

            return dist;
        };

        // size of the sliding window
        int winSize_ = 5;

        float speed_ = 0;
        std::vector<float> distances_;
        float dist_ = 0;
        std::vector<cv::Mat> images_;

        std::string loadFile;
        int frameCnt = 0;
        NeuralNetwork::FeedForward::Network net_;

        void initNN(NeuralNetwork::FeedForward::Network& n, const std::string& file);
        void loadNetFromFile(const std::string& file);
        double predictSpeedByNN(const cv::Mat flow);
        double predictDistByNN(const cv::Mat flow);

    };
}
