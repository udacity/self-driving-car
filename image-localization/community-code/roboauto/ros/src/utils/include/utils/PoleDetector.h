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
 * @brief: A class to detect pole-like objects that serve as image descriptors.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "CVSerializationHelper.h"

namespace utils {
class PoleDetector {

public:
    struct Pole {
        cv::RotatedRect bbox;
        cv::Point2f position;

    private:
        // Boost serialization
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & bbox;
            ar & position;
        }
    };

    std::vector<Pole> FindPolesInImage(const cv::Mat &img);

    void DrawPolesToImage(cv::Mat &img,const std::vector<Pole> &poles);

private:

    cv::Mat erodeStruc_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,30));
    cv::Mat closingStruc_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8,20));

    int lowThreshold = 140;
    int rat = 3;

};

}