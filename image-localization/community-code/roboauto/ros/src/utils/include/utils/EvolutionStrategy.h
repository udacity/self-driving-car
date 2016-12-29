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
 * @brief: A tool that helped us find the optimal values of certain constants.
 */

#pragma once

#include "Random.h"
#include "General.h"

#include <vector>
#include <array>
#include <limits>
#include <iostream>
#include <functional>

namespace utils {
namespace ES {
    template<std::size_t DIM = 1>
    class ES {
    public:
        class Individual {
            public:
                auto operator[](std::size_t i) const {
                    return dim[i];
                }
                std::array<float,DIM> dim;
                float fitness = FMATH_NAN;
        };

        Individual run() {
            evaluations = 0;
            std::array<float,DIM> sigma = initialSigma;
            parent = generateRandomIndividual();
            parent.fitness = computeFitness(parent);
            std::cout << "-1 : ";
            printIndividual(parent);

            for(std::size_t i=0;i<maxIterations;i++) {
                std::size_t iMaxPops = 0;
                Individual newParent;

                for(std::size_t childCount = 0;childCount < childrenCount;childCount++) {
                    Individual best;
                    for(std::size_t childrenGen=0;childrenGen<iMax;childrenGen++) {
                        Individual newChild = Mutate(parent,sigma);
                        newChild.fitness = computeFitness(newChild);
                        if(newChild.fitness < parent.fitness) {
                            iMaxPops ++;
                            if(utils::isfnan(best.fitness) || best.fitness > newChild.fitness) {
                                best=newChild;
                            }
                        }
                    }
                    if(!utils::isfnan(best.fitness) && (utils::isfnan(newParent.fitness) || newParent.fitness > best.fitness)) {
                        newParent = best;
                    }
                }

                if(!utils::isfnan(newParent.fitness)) {
                    parent = newParent;
                }

                /* sigma modification */
                if(static_cast<float>(iMaxPops)/(iMax*childrenCount) < targetPercentage) {
                    for(std::size_t i=0;i<DIM;i++) {
                        sigma[i] *= sigmaChange;
                    }
                } else if (static_cast<float>(iMaxPops)/(iMax*childrenCount) > targetPercentage) {
                    for(std::size_t i=0;i<DIM;i++) {
                        sigma[i] /= sigmaChange;
                    }
                }

                for(std::size_t i=0;i<DIM;i++) {
                    float maxSpace = _constraints[i].second - _constraints[i].first;
                    if(sigma[i] > maxSpace/2.0) {
                        sigma[i]= maxSpace/2.0;
                    }
                }

                std::cout << i << " : ";
                printIndividual(parent);

                if(parent.fitness < targetFitness) {
                    return parent;
                }
            }

            return parent;
        }

        void setFitnessFunction(std::function<float (const Individual &ind)> fun) {
            fitnessFunction = fun;
        }

        std::array<std::pair<float,float>,DIM> _constraints;
        std::array<float,DIM> initialSigma;
        std::size_t maxIterations = 50;
        float targetFitness = 1.0;
        float targetPercentage = 0.2;
        std::size_t evaluations =0;
        std::size_t iMax = 50;
        std::size_t childrenCount = 10;
    protected:
        void printIndividual (const Individual &ind) {
            std::cout << "[ " << ind.dim[0];
            for(std::size_t i=1;i<DIM;i++) {
                std::cout << ", " << ind.dim[i];
            }
            std::cout << "] : " << ind.fitness << "\n";
        }

        float computeFitness(const Individual &ind) {
            if(utils::isfnan(ind.fitness)) {
                evaluations++;
                return fitnessFunction(ind);
            }
            return ind.fitness;
        }

        Individual Mutate(const Individual &parent, const std::array<float,DIM> &sigma) const {
            Individual ret;
            for(std::size_t i=0;i<DIM;i++) {
                float gen = un();
                float newPos = parent.dim [i] + gen*sigma[i];
                while(newPos < _constraints[i].first || newPos > _constraints[i].second) {
                    if(newPos < _constraints[i].first) {
                        newPos = 2.0f*_constraints[i].first - newPos;
                    } else if(newPos > _constraints[i].second) {
                        newPos = 2.0f*_constraints[i].second -newPos;
                    }
                }
                ret.dim[i] =newPos;
            }
            return ret;
        }

        Individual generateRandomIndividual() const {
            Individual rand;
            for(std::size_t i=0;i<DIM;i++) {
                utils::Random::Uniform uniform(_constraints[i].first,_constraints[i].second);
                rand.dim[i] = uniform();
            }
            return rand;
        }

        std::function<float (const Individual &ind)> fitnessFunction;

        utils::Random::Normal un = utils::Random::Normal(0,0.5);
        float sigmaChange = 0.9;
        Individual parent;
        std::vector<Individual> _generation;
    };
}
}