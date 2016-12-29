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
 * @brief: A custom random number generator.
 */

#pragma once

#include <cstdlib>
#include <limits>
#include <cmath>

namespace utils {
    namespace Random {
        class Random {
        public:
            virtual double operator()() const = 0;
        };

        class Normal : public Random {
        public:
            Normal(const double& mean = 0.0, const double& stddev = 1.0) : mean_(mean), stddev_(stddev)
            {

            }

            virtual double operator()() const
            {
                const double epsilon = std::numeric_limits<double>::min();
                const double two_pi = 2.0 * M_PI;

                static double z0, z1;
                static bool generate;
                generate = !generate;

                if (!generate)
                    return z1 * stddev_ + mean_;

                double u1, u2;
                do {
                    u1 = rand() * (1.0 / RAND_MAX);
                    u2 = rand() * (1.0 / RAND_MAX);
                }
                while (u1 <= epsilon);

                z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
                z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
                return z0 * stddev_ + mean_;
            }

        protected:
            double mean_;
            double stddev_;
        };

        class Uniform : public Random {
        public:
            Uniform(const double& min, const double& max) : min_(min), max_(max) { }

            virtual double operator()() const
            {
                double random = static_cast<double>(rand()) / RAND_MAX;
                return min_ + random * (max_ - min_);
            }

        protected:
            double min_;
            double max_;
        };
    }
}