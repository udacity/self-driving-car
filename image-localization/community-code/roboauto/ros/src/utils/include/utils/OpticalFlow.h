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
 * @author: RoboAuto Team
 * @brief: A class that wraps OpenCV's Optical Flow algorithm.
 */

#pragma once

#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>

#include <ros/ros.h>

namespace utils {
    namespace Motion {
        inline cv::Mat getOptFlow(const cv::Mat& prev,const cv::Mat& curr, int xSampleStep, int ySampleStep) {
            int xSize = prev.cols / xSampleStep;
            int ySize = prev.rows / ySampleStep;

            cv::Mat prevGray, currGray;

            // rgb 2 gray
            if(curr.type() != 0) {
                cvtColor(curr, currGray, CV_BGR2GRAY);
            } else {
                currGray=curr;
            }
            if(prev.type() != 0) {
                cvtColor(prev, prevGray, CV_BGR2GRAY);
            } else {
                prevGray=prev;
            }

            cv::UMat Ures;
            cv::Mat flowMat;
            cv::calcOpticalFlowFarneback(prevGray, currGray, Ures, 0.4, 1, 12, 2, 8, 1.2, 0);
            cv::resize(Ures,flowMat,cv::Size(xSize,ySize),0,0,cv::INTER_NEAREST);

            return flowMat;
        };

        inline cv::Mat DrawOpticalFlow(const cv::Mat& curr,const cv::Mat& flowMat, float flowCoef = 1.0) {
            int xSampleStep = curr.cols/flowMat.cols;
            int ySampleStep = curr.rows/flowMat.rows;
            cv::Mat img;
            curr.copyTo(img);

            for (int y = 0; y < flowMat.rows; y++) {
                for (int x = 0; x < flowMat.cols; x++) {
                    cv::Point2f flow = flowMat.at<cv::Point2f>(y, x);

                    line(img,
                         cv::Point(x * xSampleStep, y * ySampleStep), cv::Point(cvRound(x * xSampleStep + flow.x*flowCoef),
                                                                                cvRound(y * ySampleStep + flow.y*flowCoef)),
                         cv::Scalar(flow.x < 0 ? 255 : 0, flow.x < 0 ? 0 : 255, 0));
                    // draw initial point
                    circle(img, cv::Point(x * xSampleStep, y * ySampleStep), 1, cv::Scalar(0, 0, 0), -1);
                }
            }
            return img;
        }
    };
}