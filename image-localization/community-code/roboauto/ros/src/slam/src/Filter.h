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
 * @author RoboAuto team
 * @brief The implementation of classes needed by particle filter - Initialization and Resampling functions.
 */

#pragma once

#include <utils/Map.h>
#include <utils/ParticleFilter.h>
#include <ImageDescriptor/Horizont.h>

namespace Slam {
    class InitializationFunction : public utils::ParticleFilter::ParticleFilter<1,0>::InitializationFunction {
    public:

        InitializationFunction(float from, float to): gen(from, to-0.001)  {
            byStep = false;
        }

        InitializationFunction(float from, float to, std::size_t particles): from_(from), step((to-from)/particles), gen(from, to-0.001)  {
            byStep = true;
        }

        void initialize(utils::ParticleFilter::ParticleFilter<1, 0>::Particle &entity) const override {
            if(byStep) {
                entity[0] = from_ + step * index;
                index++;
            }else{
                entity[0] = gen();
            }
        }

    private:
        float from_;
        float step = 0;
        mutable std::size_t index = 0;
        bool byStep;
        utils::Map::Map map_;
        utils::Random::Uniform gen;
    };

    class ResamplingFunction : public utils::ParticleFilter::ParticleFilter<1,0>::ResamplingFunction {
    public:
        ResamplingFunction(const utils::Map::Map &map, float speed, float error): un(speed,error), map_(map),speed_(speed) {

        }

        /* @brief finds exit node on segment, if no exit exists (last segment) -1 is returned */
        int GetExitNodeOnSegment(std::size_t segmentID, double coef) const {
            /* first find exit node of segment */
            int exitNodeOnSegment = -1;
            std::size_t numberOfSegments = 0;
            for (const auto &a: map_.GetSegment(segmentID).getNextSegments()) {
                if (std::get<0>(a) >= coef) {
                    numberOfSegments++;
                }
            }

            if(numberOfSegments > 0) {
                std::size_t exitSegmentNumber = rand()%numberOfSegments;
                for(std::size_t i=0;i<map_.GetSegment(segmentID).getNextSegments().size();i++) {
                    const auto &a = map_.GetSegment(segmentID).getNextSegments()[i];
                    if (std::get<0>(a) >= coef) {
                        if(exitSegmentNumber == 0) {
                            return static_cast<int>(i);
                        }
                        exitSegmentNumber--;
                    }
                }
            }
            return exitNodeOnSegment;
        }

        void Resample(utils::ParticleFilter::ParticleFilter<1, 0>::Particle &entity) const override {
            double position = entity[0];
            std::size_t segmentID = static_cast<std::size_t>(floor(position));
            double coef = position - floor(position);
            double distanceToShift = un();

            if(distanceToShift < 0) {
                distanceToShift =fabs(distanceToShift);
                double onSegment = cv::norm(map_.GetSegment(segmentID).GetDir()*coef);
                if(onSegment < distanceToShift) {
                    distanceToShift = onSegment;
                }
                coef-= distanceToShift/map_.GetSegment(segmentID).GetSize();

                distanceToShift= 0.0;
            }

            while(distanceToShift > 0.0) {
                int exitNode = GetExitNodeOnSegment(segmentID, coef);
                double exitCoef = exitNode >= 0 ? std::get<0>(map_.GetSegment(segmentID).getNextSegments()[exitNode]) : 1.0;

                double remainingOnSegment = cv::norm(map_.GetSegment(segmentID).GetDir()*(exitCoef - coef));

                // Stay on segment
                if(remainingOnSegment > distanceToShift) {
                    coef += distanceToShift / cv::norm(map_.GetSegment(segmentID).GetDir());
                    distanceToShift = 0.0;
                } else if (exitNode >= 0) { // There is next segment
                    distanceToShift -= remainingOnSegment;
                    coef = std::get<2>(map_.GetSegment(segmentID).getNextSegments()[exitNode]);
                    segmentID = std::get<1>(map_.GetSegment(segmentID).getNextSegments()[exitNode]);
                } else { // We are in the last segment
                    coef = 0.999;
                    distanceToShift = 0.0;
                }
            }

            entity[0] = static_cast<double>(segmentID)+coef;
        }

        void Normalize(utils::ParticleFilter::ParticleFilter<1, 0>::Particle &entity) const override {
            if(floor(entity[0]) >= map_.GetSegmentsSize()) {
                entity[0] = map_.GetSegmentsSize()-0.001;
            } else if(entity[0] < 0) {
                entity[0] = 0;
            }
        }

        utils::Random::Normal un;
        const utils::Map::Map &map_;
        float speed_;
    };

};