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
 * @brief: A class used for tweaking some parameters using evolution strategies.
 */

#include <utils/EvolutionStrategy.h>
#include "TestSlam.h"

int main(int argc, char** argv) {

    srand(time(NULL));

    /* DIMENSIONS
     * 0 : horizont
     * 1 : polesPositive
     * 2 : polesNegative
     * 3 : signsPositive
     * 4 : signsNegative
     * 5 : lightsPositive
     * 6 : lightsNegative
     */

    using Individual =utils::ES::ES<5>::Individual;
    utils::ES::ES<5> evol;
    // horizont
    float sigma = 0.05;
    evol.initialSigma[0] = sigma;
    evol._constraints[0].first = -0.5;
    evol._constraints[0].second = -0.0001f;
    // poles positive
    evol.initialSigma[1] = sigma;
    evol._constraints[1].first = 0.5f;
    evol._constraints[1].second = 0.99;
    // poles negative
    evol.initialSigma[2] = sigma;
    evol._constraints[2].first = 0.5f;
    evol._constraints[2].second = 0.99;
    // signs positive
    evol.initialSigma[3] = sigma;
    evol._constraints[3].first = 0.5f;
    evol._constraints[3].second = 0.99;
    // signs positive
    evol.initialSigma[4] = sigma;
    evol._constraints[4].first = 0.5f;
    evol._constraints[4].second = 0.99;
/*    // lights positive
    evol.initialSigma[5] = sigma;
    evol._constraints[5].first = 0.5f;
    evol._constraints[5].second = 0.99;
    // lights positive
    evol.initialSigma[6] = sigma;
    evol._constraints[6].first = 0.5f;
    evol._constraints[6].second = 0.99;
*/
    evol.targetFitness = 0.000001;
    evol.maxIterations = 1000;
    evol.iMax=10;
    evol.childrenCount=3;

    const auto runs = 1;
    std::function<float (const Individual &ind)> fitness = [] (const Individual &ind) ->float {
        Slam::TestSlam slam;
        Slam::WeightFunction::Config config;
        config.horizontParam = ind[0];
        config.polesPositive = ind[1];
        config.polesNegative = ind[2];
        config.signsPositive = ind[3];
        config.signsNegative = ind[4];
        slam.SetConfig(config);
        //slam.weightFunctionConfig_.horizontParam = ind[5];
        //slam.weightFunctionConfig_.horizontParam = ind[6];
        slam.LoadMap("/home/robo/map.data");
        slam.LoadRun("/home/robo/run.data");

        float score = 0;
        for(int i=0;i<runs;i++) {
           score += slam.TestRun().first;
        }
        std::cout << score/runs << "\n";
        return score/runs;

        /*float fitness = fabs(ind[0]+0.25) + fabs(ind[1]-0.7) + fabs(ind[2]-0.7) + fabs(ind[3]-0.7) + fabs(ind[4]-0.7) + fabs(ind[5]-0.7) + fabs(ind[6]-0.7);

        return fitness;*/
    };

    evol.setFitnessFunction(fitness);
    evol.run();

    std::cout << evol.evaluations << "\n";

    //ros::spin();
}