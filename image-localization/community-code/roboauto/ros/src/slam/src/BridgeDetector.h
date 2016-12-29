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
 * @brief: A Class for bridge detection.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <limits>

#include <utils/Camera.h>

#include <utils/Topics.h>

#include <ros/ros.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>


namespace BridgeDetector {
    class BridgeDetector{
    public:

        BridgeDetector() : rng(12345) {};

        bool findBridge(const cv::Mat& img){
            cv::Mat bgr;
            cvtColor(img, bgr, CV_BGR2GRAY);

            cv::Mat thresh;
            threshold(bgr, thresh, 100, 255, cv::THRESH_BINARY);

            cv::Mat dilated;
            dilate(thresh, dilated, dilElem);
            erode(dilated, dilated, dilElem);
            dilate(dilated, dilated, dilElem);

            // crop sides and bottom
            cv::Mat top = dilated(cv::Rect(cropLeft,0,dilated.cols-(cropLeft + cropRight),4*dilated.rows/10.));
            std::vector<cv::Vec4i> hierarchy;
            std::vector<std::vector<cv::Point>> contours;

            cv::Mat contourmat;
            top.copyTo(contourmat);
            // white border - detect border as edge
            //              src,       dst,     top, bott, left, right
            copyMakeBorder(contourmat,contourmat,1,0,1,1,cv::BORDER_CONSTANT,255);
            Canny( contourmat, contourmat, 50, 150, 3 );
            // close gaps in the contour
            morphologyEx(contourmat,contourmat,cv::MORPH_CLOSE,closingStruc_);

            findContours( contourmat, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

            std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
            std::vector<cv::Rect> boundRect( contours.size() );

            for( std::size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 1, true );
                boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
            }

            cv::Mat drawing;
            img.copyTo(drawing);

            std::pair<int, int> ret;
            ret.first = -1;
            ret.second = -1;

            for( std::size_t i = 0; i< contours.size(); i++ ) {
                // needs to be over whole width
                if (boundRect[i].width < (top.cols - 5)) {
                    continue;
                }

                // too high
                if (boundRect[i].height > (8 * top.rows / 10.)){
                    continue;
                }

                // not in top
                if(abs(boundRect[i].tl().x) > 5
                   || abs(boundRect[i].tl().y) > 5){
                    // too high and not in top
                    if(boundRect[i].height > (3*top.rows/4.)) {
                        continue;
                    }
                    // too close to bottom
                    if(abs(boundRect[i].br().y - top.rows) < 5) {
                        continue;
                    }
                }

                // Bridge positive!
                ret.first = boundRect[i].tl().y;
                ret.second = boundRect[i].height;

                cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point() );
                boundRect[i].width += cropLeft + cropRight;
                rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
            }

            if(ret.first < 0)
                return false;
            return true;
        };
    private:
        cv::RNG rng;

        // more sensitive - more false positives
        cv::Mat dilElem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(150, 8));
        cv::Mat closingStruc_ = getStructuringElement(cv::MORPH_RECT, cv::Size(15,15));
        // crop sides in px
        int cropLeft = 100;
        int cropRight = 100;
    };
}
