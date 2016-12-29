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
 * @brief: A class that serves as a traffic light detector; implemented using OpenCV's blob detection.
 */

#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "utils/General.h"

class TrafficLightDetector {
    using Img = cv::Mat;
    cv::Ptr<cv::SimpleBlobDetector> blobDetector_;

    struct BlobPoint {

        cv::Point pt;
        cv::Vec3b clr;

        BlobPoint() : pt{}, clr{}
        {}

        BlobPoint(const BlobPoint& _kp): pt(_kp.pt), clr(_kp.clr)
        {
        }

        BlobPoint(BlobPoint&& _kp): pt(std::move(_kp.pt)), clr(std::move(_kp.clr))
        {
        }

        BlobPoint(cv::KeyPoint&& _kp, const cv::Vec3b& _clr) : pt(std::move(_kp.pt)), clr(_clr)
        {
        }

        BlobPoint(const cv::KeyPoint& _kp, const cv::Vec3b& _clr) : pt(_kp.pt), clr(_clr)
        {
        }

        BlobPoint& operator=(BlobPoint&& _kp)
        {
            pt = std::move(_kp.pt);
            clr = std::move(_kp.clr);
            return *this;
        }

        BlobPoint& operator=(const BlobPoint& _kp)
        {
            pt = _kp.pt;
            clr = _kp.clr;
            return *this;
        }

        bool operator==(const BlobPoint& p)
        {
            return this->pt == p.pt && this->clr == p.clr;
        }
    };

    std::vector<cv::Point> BlobPtToCVPt(const std::vector<BlobPoint>& bPts)
    {
        std::vector<cv::Point> points;
        for (const auto& bPt : bPts) {
            points.push_back(bPt.pt);
        }
        return points;
    }

    void SetBlobDetParams(cv::SimpleBlobDetector::Params& blobParams)
    {
        blobParams.minThreshold = 0;
        blobParams.maxThreshold = 256;

        blobParams.minDistBetweenBlobs = 10;

        blobParams.minRepeatability = 2;

        blobParams.filterByArea = true;
        blobParams.minArea = 35;
        blobParams.maxArea = 220;

        blobParams.filterByInertia = false;

        blobParams.filterByConvexity = false;

        blobParams.filterByColor = true;
        blobParams.blobColor = 255;

        blobParams.filterByCircularity = true;
        blobParams.minCircularity = 0.7f;
        blobParams.maxCircularity = std::numeric_limits<float>::max();
    }

    bool TLColour(const cv::Vec3b& clr)
    {
        return (clr[1] == 255 && (clr[0] < 145 || clr[2] < 145))
            || (clr[2] == 255 && (clr[0] < 145 || clr[1] < 145));
    }

    std::vector<BlobPoint> lastFramesBrakePoints_;
    const uint8_t maxLastFramePoints = 19;
    std::vector<BlobPoint> RemoveBrakeLights(std::vector<BlobPoint>&& blobPoints)
    {
        std::vector<BlobPoint> noBrakes, brakes;
        for (auto& p1 : blobPoints) {
            bool pointNearby = false;
            for (const auto& p2 : blobPoints) {
                if (p1 == p2 || p1.clr[2] != 255 || p2.clr[2] != 255) {
                    continue;
                }
                if (std::abs(p1.pt.x - p2.pt.x) < 45 && std::abs(p1.pt.y - p2.pt.y) < 8) {
                    pointNearby = true;
                    brakes.push_back(std::move(p2));
                }
            }
            for (const auto& p2 : lastFramesBrakePoints_) {
                if (p1.clr[2] != 255 || p2.clr[2] != 255) {
                    continue;
                }
                if (std::abs(p1.pt.x - p2.pt.x) < 45 && std::abs(p1.pt.y - p2.pt.y) < 8) {
                    pointNearby = true;
                    brakes.push_back(std::move(p1));
                }
            }
            if (!pointNearby) {
                noBrakes.push_back(std::move(p1));
            }
        }

        lastFramesBrakePoints_.insert(lastFramesBrakePoints_.end(), brakes.cbegin(), brakes.cend());

        if (lastFramesBrakePoints_.size() > maxLastFramePoints) {
            auto overFlow = lastFramesBrakePoints_.size() - maxLastFramePoints;
            std::rotate(lastFramesBrakePoints_.begin(), lastFramesBrakePoints_.begin() + overFlow ,lastFramesBrakePoints_.end());
            lastFramesBrakePoints_.resize(maxLastFramePoints);
        }

        return noBrakes;
    }

    std::vector<BlobPoint> FilterTrafficLights(std::vector<BlobPoint>&& blobPoints)
    {
        std::vector<BlobPoint> tLights;
        for (auto&& kp : blobPoints) {
            if (TLColour(kp.clr)) {
                tLights.push_back(std::move(kp));
            }
        }
        return RemoveBrakeLights(std::move(tLights));
    }

public:
    TrafficLightDetector()
    {
        cv::SimpleBlobDetector::Params blobParams;
        SetBlobDetParams(blobParams);
        blobDetector_ = cv::SimpleBlobDetector::create(blobParams);
    }

    std::vector<cv::Point> FindTrafficLights(const Img& img)
    {
        Img croppedImg = img(cv::Rect{0, 0, img.cols, static_cast<int>(img.rows * 0.45)});

        std::vector<cv::KeyPoint> keyPoints;
        blobDetector_->detect(croppedImg, keyPoints);

        std::vector<BlobPoint> blobPoints;
        for (auto&& kp : keyPoints) {
            blobPoints.emplace_back(std::move(kp), croppedImg.at<cv::Vec3b>(kp.pt));
        }

        auto trafficLights = FilterTrafficLights(std::move(blobPoints));

        return BlobPtToCVPt(trafficLights);
    }

    void DisplayTraficLights(Img& img, const std::vector<cv::Point>& points) {
        for (const auto& kp : points) {
            cv::circle(img, kp, 6, {255, 255, 255}, 3);
        }
    }
};


