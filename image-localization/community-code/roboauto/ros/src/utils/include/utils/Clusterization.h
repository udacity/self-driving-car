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
 * @brief: A class that implements a clusterization algorithm using weighted point density and distance.
 */

#pragma once

#include "ParticleFilter.h"
#include "Map.h"

#include <vector>

namespace utils {
    class Clusterization {

    public:
        Clusterization(Map::Map& map) : map_(map)
        {

        }

        void Reset()
        {
            particleVec_.clear();
            nearest_.clear();
        }

        double SURROUNDING_METERS = 25;

        void Update(const std::vector<utils::ParticleFilter::ParticleFilter<1,0>::Particle>& particles)
        {
            particleVec_.resize(particles.size());
            nearest_.resize(particles.size());
            for(std::size_t i = 0; i < particles.size(); i++) {
                particleVec_[i] = particles[i][0];
            }

            std::sort(particleVec_.begin(), particleVec_.end(), [](const double& l, const double& r) {return l < r;});

            for(std::size_t i = 0; i < particleVec_.size(); i++) {
                const auto segment = particleVec_[i];
                double surrounding = std::max(std::min(SURROUNDING_METERS / map_.GetSegment(static_cast<std::size_t>(floor(segment))).GetSize(), 5.0), 0.2);
                double min = segment - surrounding;
                double max = segment + surrounding;

                std::size_t firstIndex = std::lower_bound(particleVec_.begin(), particleVec_.begin()+i,min) - particleVec_.begin();
                std::size_t afterIndex = std::upper_bound(particleVec_.begin()+i, particleVec_.end(),max) - particleVec_.begin();
                double score = 0;
                for(std::size_t j=firstIndex;j<afterIndex;j++) {
                    double dist = std::max(map_.GetDistBetweenSegments(segment,particleVec_[j]),0.01);
                    score+=1.0/dist;
                }
                nearest_[i]= score; //afterIndex-firstIndex;
            }
        }

        std::pair<double,double> GetBestCluster() const
        {
            auto result = std::max_element(nearest_.begin(), nearest_.end());
            return {particleVec_[result - nearest_.begin()], nearest_[result - nearest_.begin()] };
        }

        std::vector<std::pair<double, std::size_t>> GetClusters() const
        {
            std::vector<std::pair<double, std::size_t>> clusters;

            auto maxIter = std::max_element(nearest_.begin(), nearest_.end());
            std::size_t maxSize = nearest_[maxIter - nearest_.begin()];
            std::size_t minimalSize = maxSize / 1.5;

            for(std::size_t i = 0; i < particleVec_.size(); i++) {
                if(nearest_[i] > minimalSize) {
                    clusters.push_back({particleVec_[i], nearest_[i]});
                }
            }

            return clusters;
        }

        std::vector<std::pair<double, std::size_t>> GetAllClusters() const
        {
            std::vector<std::pair<double, std::size_t>> clusters;

            for(std::size_t i = 0; i < particleVec_.size(); i++) {
                clusters.push_back({particleVec_[i], nearest_[i]});
            }

            return clusters;
        }
    protected:
        Map::Map& map_;
        std::vector<double> particleVec_;
        std::vector<double> nearest_;
    };
}