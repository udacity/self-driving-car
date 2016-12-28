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
 * @brief: A ros node module used for statistical evaluation of multiple runs.
 */

#include <utils/EvolutionStrategy.h>
#include "TestSlam.h"

int main(int argc, char** argv) {

    std::string mapFile = argv[1];
    std::string dataFile = argv[2];

    Slam::TestSlam slam;
    slam.LoadMap(mapFile);
    slam.LoadRun(dataFile);

    std::size_t runs = 100;

    float score = 0;
    for(std::size_t i=0;i<runs;i++) {
        auto seed = time(NULL);
        std::cout << seed << "\n";
        srand(seed);//time(NULL));
        auto error =slam.TestRun();
        std::cout << i << ": " << error.first << ", " << error.second  << "/ " << slam.runSamples_.size() << "\n";
        score += error.first;
    }
    std::cout << score/runs << "\n";
    return score/runs;
}