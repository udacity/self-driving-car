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
 * @brief: A class containing a learning sample for the motion prediction.
 */

#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Config.h"

namespace Sample{

    using Sample = std::pair<std::vector<float>, std::vector<float>>;

    __attribute__((unused)) static std::vector<float> createSample(cv::Mat flow){
        std::vector<float> in;
        cv::Point2f val;
        for(int y=0; y<flow.rows; y++){
            for(int x=0; x<flow.cols; x++){
                val = flow.at<cv::Point2f>(y,x);
                const double fx = val.x;
                const double fy = val.y;

                // cropped to [-100, 100] range, then normalized
                in.push_back(static_cast<float>(std::min(std::max(fx,-Motion::MAX_FLOW),Motion::MAX_FLOW) / Motion::MAX_FLOW));
                in.push_back(static_cast<float>(std::min(std::max(fy,-Motion::MAX_FLOW),Motion::MAX_FLOW) / Motion::MAX_FLOW));
            }
        }

        return in;
    };

    // optimalize return?
    __attribute__((unused)) static Sample createSample(cv::Mat flow, double dist){
        std::vector<float> in = createSample(flow);
        std::vector<float> speedv;

        // speed normalized to [0,1]
        speedv.push_back(static_cast<float>(std::min(dist,Motion::MAX_DIST) / Motion::MAX_DIST));

        std::cout << "Dist: " << dist << " Normalized: " << speedv[speedv.size()-1] << std::endl;

        return std::make_pair(in, speedv);
    };
}