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
 * @brief: A tool for visual color filtration in the HSV colorspace. Uses a simple trackbar for selecting displayed colors.
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <utils/OpticalFlow.h>


const int alpha_slider_max = 255;
int fromH;
int fromS;
int fromV;

int toH = 255;
int toS = 255;
int toV = 255;

int elementSize=0;
int erodeSize=0;
cv::Mat hsvImg;

void on_trackbar( int, void* )
{
    cv::Mat marks;
    cv::inRange(hsvImg, cv::Scalar(fromH,fromS, fromV), cv::Scalar(toH, toS, toV), marks);
    if(erodeSize > 0 ) {
        cv::Mat element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erodeSize, erodeSize));
        erode(marks, marks, element_);
    }
    if(elementSize > 0) {
        cv::Mat element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(elementSize, elementSize));
        dilate(marks, marks, element_);
    }
    cv::imshow("Selector", marks);
}

int lastKey;

namespace utils {
    class ColorFiltration {
    public:
        static void ColorSelector (const cv::Mat& h) {
            hsvImg=h;
            cv::namedWindow("Selector", 1);
            cv::createTrackbar( "From H", "Selector", &fromH, alpha_slider_max, on_trackbar );
            cv::createTrackbar( "From S", "Selector", &fromS, alpha_slider_max, on_trackbar );
            cv::createTrackbar( "From V", "Selector", &fromV, alpha_slider_max, on_trackbar );

            cv::createTrackbar( "To H", "Selector", &toH, alpha_slider_max, on_trackbar );
            cv::createTrackbar( "To S", "Selector", &toS, alpha_slider_max, on_trackbar );
            cv::createTrackbar( "To V", "Selector", &toV, alpha_slider_max, on_trackbar );
            cv::createTrackbar( "DilateteSize", "Selector", &elementSize, alpha_slider_max, on_trackbar );
            cv::createTrackbar( "ErodeS", "Selector", &erodeSize, alpha_slider_max, on_trackbar );
            on_trackbar(0,0);
            int key;
            do{
                if(lastKey == 'p') {
                    key = cv::waitKey(1);
                    if(key != -1) {
                        lastKey = key;
                    }
                } else {
                    lastKey = cv::waitKey(0);
                }

            } while(lastKey != 'p');
        }
    };
}