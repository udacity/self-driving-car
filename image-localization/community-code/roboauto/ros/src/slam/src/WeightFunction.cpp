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

#include "WeightFunction.h"

/// all the values for scoring functions are loaded from a config file.



double Slam::WeightFunction::GetScore(std::size_t segment, double percentage) {
    double h, p, ys, dgs, dys, l, b;
    double score;
    std::size_t numberOfPoints = map_.GetSegment(segment).GetMapPoints().size();
    std::size_t computedPoint = percentage * numberOfPoints;

    h = getHorizontScore(segment, computedPoint);
    l = getLightScore(map_.GetSegment(segment), computedPoint, curr_.lights);
    ys = getSignsScore(map_.GetSegment(segment).GetMapPoints()[computedPoint].detectedObjecsts.yellowSigns, curr_.yellowSigns);
    dgs = getSignsScore(map_.GetSegment(segment).GetMapPoints()[computedPoint].detectedObjecsts.darkGreenSigns, curr_.darkGreenSigns);
    dys = getSignsScore(map_.GetSegment(segment).GetMapPoints()[computedPoint].detectedObjecsts.darkYellowSigns, curr_.darkYellowSigns);
    p = getPolesScore(segment, computedPoint);
    b = getBridgeScore(map_.GetSegment(segment).GetMapPoints()[computedPoint].detectedObjecsts.bridge, curr_.bridge);

    score = h * l * ys * dgs * dys * p * b;

    return score;
}

double Slam::WeightFunction::getHorizontScore(std::size_t posA, std::size_t mapPoint) const {
    double score = desriptor_.Compare(map_.GetSegment(posA).GetMapPoints()[mapPoint].detectedObjecsts.horizont, curr_.horizont);
    return exp(config_.horizontParam * score);
}

double Slam::WeightFunction::getPolesScore(std::size_t pos, std::size_t mapPoint) const {
    const auto&poles = map_.GetSegment(pos).GetMapPoints()[mapPoint].detectedObjecsts.poles;
    const auto& currPoles = curr_.poles;

    double score = 1.0;

    float closestDistX = std::numeric_limits<float>::max();
    for (size_t i = 0; i < currPoles.size(); ++i)
    {
        //closest pole
        for (size_t j = 0; j < poles.size(); ++j)
        {
            float distX = std::fabs(currPoles[i].position.x - poles[j].position.x);
            if (distX < closestDistX){
                closestDistX = distX;
            }
        }

        if (closestDistX < 150){
            score *= config_.polesPositive;
        }

        closestDistX = std::numeric_limits<float>::max();
    }

    return score;
}

double Slam::WeightFunction::getSignsScore(const std::vector<utils::SignDetection::Sign> &dbSigns, const std::vector<utils::SignDetection::Sign> & currSigns) const {
    int found = 0;
    for(const auto& sign: dbSigns) {
        double nearest = std::numeric_limits<float>::max();
        for(const auto& currSign: currSigns) {
            double norm = cv::norm(currSign.first - sign.first);
            if(norm < nearest) {
                nearest = norm;
            }
        }
        if(nearest < 200.0) {
            found++;
        }
    }
    return pow(config_.signsPositive,found) * pow(config_.signsNegative,dbSigns.size() - found);
}

double Slam::WeightFunction::getLightScore(const utils::Map::Segment &segment,const size_t computedPoint, const std::vector<cv::Point> & lights) const {
    if (lights.empty()) {
        return config_.lightsNegative;
    }

    return segment.GetMapPoints()[computedPoint].detectedObjecsts.semaphore ? config_.lightsPositive : config_.lightsNegative;
}

double Slam::WeightFunction::getBridgeScore(bool segBridge, bool currBridge) const {
    if(segBridge)
        return segBridge == currBridge ? config_.bridgePositive : config_.bridgeNegative;
    return config_.bridgeNegative;
}