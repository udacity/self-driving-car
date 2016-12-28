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
 * @brief: Methods that implement a basic detection of sign-like objects we use as image descriptors.
*/

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <utils/OpticalFlow.h>

namespace utils {
    namespace SignDetection {
        using Sign = std::pair<cv::Point2f, float>;

        std::vector<Sign> FindSigns(cv::Mat &marks, int radius = 10);

        void DrawSignToImg(cv::Mat &img, const std::vector<Sign>& marks, const std::string &type = "");

        std::vector<Sign> DetectYellowMarks(const cv::Mat& hsvImage);

        std::vector<Sign> DetectDarkYellowMarks(const cv::Mat& hsvImage);

        std::vector<Sign> DetectDarkGreen(const cv::Mat& hsvImage);
    };
}