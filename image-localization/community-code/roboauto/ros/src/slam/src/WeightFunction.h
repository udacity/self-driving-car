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
 * @brief: A class containing calculations of a score for a given position in the map.
 *
 * A given position in the map is compared to the descriptors of a actual frame and it is calculated its score.
 * The score of a position is put together from individual scores of each descriptor.
 */

#pragma once

#include <utils/Map.h>
#include <utils/ParticleFilter.h>
#include <ImageDescriptor/Horizont.h>

namespace Slam {
    class WeightFunction : public utils::ParticleFilter::ParticleFilter<1, 0>::WeightFunction {
    public:
        struct Config {
			Config() {};

            double horizontParam = -0.02;
            double polesPositive = 1.01;
            double polesNegative = 1.0;
            double signsPositive = 1.0;
            double signsNegative = 1.0;
            double lightsPositive = 1.0;
            double lightsNegative = 1.0;
            double bridgePositive = 1.0;
            double bridgeNegative = 1.0;
        };

        WeightFunction(const utils::Map::Map &map, ImageDescriptor::Horizont &desriptor, const utils::DetectedObjects &curr, Config config = {}) :
				config_(config), map_(map), desriptor_(desriptor), curr_(curr) {}

        double getHorizontScore(std::size_t posA, std::size_t mapPoint) const;
        double getLightScore(const utils::Map::Segment &segment, const size_t computedPoint, const std::vector<cv::Point> & lights) const;
        double getPolesScore(std::size_t pos, std::size_t mapPoint) const;
        double getSignsScore(const std::vector<utils::SignDetection::Sign> &dbSigns, const std::vector<utils::SignDetection::Sign> & currSigns) const;
        double getBridgeScore(bool segBridge, bool currBridge) const;

        double GetScore(std::size_t segment, double percentage);

        double operator()(const utils::ParticleFilter::ParticleFilter<1, 0>::Particle &entity) override {
            const double pos = entity[0];
            int posA = static_cast<int>(floor(pos));
            double segmentPercentage = pos - static_cast<float>(posA);

			posA = std::max(posA, 0);

            if (static_cast<std::size_t>(posA) >= map_.GetSegmentsSize())
                posA = static_cast<int>(map_.GetSegmentsSize()) - 1;

            return GetScore(static_cast<size_t>(posA),segmentPercentage);
        }


        Config config_;

        const utils::Map::Map &map_;
        ImageDescriptor::Horizont &desriptor_;
        const utils::DetectedObjects &curr_;
    };
}