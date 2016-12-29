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
 * @brief: A base class for image descriptors (actaully just for a horizon descriptor).
 */

#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ImageDescriptor {

    using Img = cv::Mat;

class ImageDescriptor {

public:
    ImageDescriptor(){
    };

    static double medianMat(Img input){
        input = input.reshape(0,1); // spread Input Mat to single row
        std::vector<double> vecFromMat;
        input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
        std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
        return vecFromMat[vecFromMat.size() / 2];
    }

    Img ExtractHorizont(const Img &img) {
        Img greyScaleImg;
        cv::cvtColor(img, greyScaleImg, CV_BGR2GRAY);
        Img tresholded;
        greyScaleImg.copyTo(tresholded);
        double median = medianMat(tresholded);
        threshold(greyScaleImg, tresholded, median * 4.5, 255, 0);
        return tresholded;
    };

    Img ExtractHorizontByColor(const Img &img) {
        Img ch[3], img_gray;
        cv::split(img, ch);
        cv::threshold(ch[0], img_gray, 100, 255, 0);

        return img_gray;
    };

    Img ExtractHorizontByOtsu(const Img &img) {
        Img ch[3], img_gray;
        cv::split(img, ch);
        cv::threshold(ch[0], img_gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

        return img_gray;
    };

};
}