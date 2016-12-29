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
 * @brief: A class for statistical evaluation of individual runs.
 *
 * For fast evaluation a map and a run dataset created by the dataSetCreator module are needed. Also @see validator.cpp.
 */

#pragma once

#include "SlamImpl.h"

#include <map>

namespace Slam {
    class TestSlam : public SlamImpl {
    public:
        std::pair<float,std::size_t> TestRun() {
            particleFilterInitialized_=false;
            Reset();
            float error = 0;
            std::size_t numbers_located = 0;
            int counter = 0;
            for(auto& runSample: runSamples_) {
                state_ = std::get<0>(runSample);
                auto coords = Localize();
                float currentError=std::pow(coords.first.DistanceTo(std::get<1>(runSample)),2);
                error+=currentError;
                if(currentError < 900) {
                    numbers_located++;
                } else if (currentError > 200*200) {
                    std:: cout << "Error " << sqrt(currentError) << ": segment " << counter << std::endl;
                }
                counter++;
            }
            return {sqrt(error/runSamples_.size()),numbers_located};
        }

        void LoadRun(const std::string &filename) {
            std::ifstream ifs(filename);
            boost::archive::binary_iarchive ia(ifs);
            ia >> runSamples_;

            for(auto& runSample: runSamples_) {
                auto& state = std::get<0>(runSample);
                state.detectedObjects.horizont.maximums = ImageDescriptor::Horizont::detectMax(state.detectedObjects.horizont.horizont);
            }
        }

        void SetConfig(const WeightFunction::Config &config) {
            weightFunctionConfig_=config;
        }

        std::vector<std::tuple<CurrentState, utils::GpsCoords>> runSamples_;
    protected:
    };
}
