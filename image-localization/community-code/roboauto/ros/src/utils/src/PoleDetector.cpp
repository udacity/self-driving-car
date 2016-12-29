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

#include <utils/PoleDetector.h>

std::vector<utils::PoleDetector::Pole> utils::PoleDetector::FindPolesInImage(const cv::Mat& img)
{
    cv::Mat res;
    cv::Canny(img,res,lowThreshold, lowThreshold*rat,3,true);
    cv::dilate(res,res, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::erode(res,res, erodeStruc_);
    cv::dilate(res,res, erodeStruc_);
    cv::morphologyEx(res,res,cv::MORPH_CLOSE,closingStruc_);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(res,contours,cv::RetrievalModes::RETR_LIST ,cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    std::vector<PoleDetector::Pole> retVal;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        Pole pole;
        pole.bbox = cv::minAreaRect(contours[i]);

        cv::Moments moments = cv::moments(contours[i],true);
        pole.position =  cv::Point2f(moments.m10/moments.m00 , moments.m01/moments.m00 );

        if ((pole.bbox.size.height/pole.bbox.size.width > 4.0 || pole.bbox.size.width/pole.bbox.size.height > 4.0)
             && (pole.bbox.size.height > 70 || pole.bbox.size.width > 70))
        {
            retVal.push_back(pole);
        }
    }

    return retVal;
}

void utils::PoleDetector::DrawPolesToImage(cv::Mat &img,const std::vector<utils::PoleDetector::Pole> &poles)
{
    for (const auto& pole : poles) {
        cv::rectangle(img,pole.bbox.boundingRect(),cv::Scalar(0,255,0),2);
        cv::circle(img,pole.position,3,cv::Scalar(0,100,255),2);
    }
}
