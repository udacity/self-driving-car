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

#include <utils/SignDetection.h>

std::vector<utils::SignDetection::Sign> utils::SignDetection::FindSigns(cv::Mat &marks, int minRadius) {

    std::vector<std::vector<cv::Point>> contours,goodContours;
    cv::findContours(marks, contours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE, cv::Point(0, 0));

    cv::Mat goodMarks = cv::Mat::zeros(marks.rows,marks.cols,marks.type());
    for (size_t i = 0; i < contours.size(); ++i)
    {
        if(contours[i].size() > 100 && contours[i].size() < 6000)
        {
            goodContours.push_back(contours[i]);
        }
    }
    cv::fillPoly(goodMarks,goodContours,cv::Scalar(255));
    contours.clear();

    cv::morphologyEx(goodMarks,goodMarks,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(7,7)));
    cv::morphologyEx(goodMarks,goodMarks,cv::MORPH_OPEN,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

    cv::findContours(goodMarks, contours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_NONE, cv::Point(0, 0));

    std::vector<Sign> detectedMarks;

    for (size_t i = 0; i < contours.size(); i++) {
        float radius;
        cv::Point2f center;
        std::vector<cv::Point> contours_poly;
        cv::approxPolyDP(cv::Mat(contours[i]), contours_poly, 3, true);
        cv::minEnclosingCircle((cv::Mat) contours_poly, center, radius);
        if (radius > minRadius) {
            detectedMarks.push_back({center, radius});
        }
    }
    return detectedMarks;
}

void utils::SignDetection::DrawSignToImg(cv::Mat &img, const std::vector<std::pair<cv::Point2f, float>> &marks,
                                         const std::string &type) {
    cv::Scalar color;
    if (type == "yellow") {
        color = cv::Scalar(21, 178, 0);
    } else if (type == "dark yellow") {
        color = cv::Scalar(219, 198, 41);
    } else if (type == "dark green") {
        color = cv::Scalar(40, 131, 22);
    } else {
        color = cv::Scalar(255, 0, 0);
    }

    for (std::size_t i = 0; i < marks.size(); i++) {
        cv::circle(img, marks[i].first, marks[i].second, color, 2, 8, 0);
    }
}

std::vector<utils::SignDetection::Sign> utils::SignDetection::DetectYellowMarks(const cv::Mat &hsvImage) {
    cv::Mat marks;
    cv::inRange(hsvImage, cv::Scalar(20, 107, 53), cv::Scalar(56, 182, 255), marks);
    return FindSigns(marks);
}

std::vector<utils::SignDetection::Sign> utils::SignDetection::DetectDarkYellowMarks(const cv::Mat &hsvImage) {
    cv::Mat marks;
    cv::inRange(hsvImage, cv::Scalar(9, 115, 64), cv::Scalar(38, 200, 200), marks);
    return FindSigns(marks);
}

std::vector<utils::SignDetection::Sign> utils::SignDetection::DetectDarkGreen(const cv::Mat &hsvImage) {
    cv::Mat marks;
    cv::inRange(hsvImage, cv::Scalar(91, 113, 32), cv::Scalar(103, 213, 208), marks);
    return FindSigns(marks, 20);
}