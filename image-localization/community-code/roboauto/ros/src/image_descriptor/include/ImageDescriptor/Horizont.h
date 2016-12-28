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
 * @brief: A class used for description and comparison of the horizons.
 */


#pragma once

#include "ImageDescriptor.h"
#include <algorithm>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

namespace ImageDescriptor {
    class Horizont : public ImageDescriptor {
    public:
        struct Description {
            std::vector<uint16_t> horizont;
            std::vector<uint16_t> horizontInverse;
            std::vector<uint16_t> maximums;
        private:
            // Boost serialization
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive & ar, const unsigned int version) {
                ar & horizont;
                ar & horizontInverse;
            }
        };

        struct Config {
            Config(){};

            int rRange = 20;
            int rMaxJump = 5;
            double rCoef = 1.0;
        };

        static struct Config config_;

        // interface function for detecting maximal elements (for compability with slamimlp and testslam)
        static std::vector<uint16_t> detectMax(const std::vector<uint16_t> &d);
        // implementations
        static std::vector<uint16_t> detectMaxPeak(const std::vector<uint16_t> &d);
        static std::vector<uint16_t> detectMaxDifference(const std::vector<uint16_t> &d);

        Description DescribeImage(const Img &img);

        void descToIMG(cv::Mat& img, const Description& desc);
        cv::Mat compareToIMG(const Description& l, const Description& r, const int shift,const int shiftY);

        int getVerticalShift(const Description& l, const Description &curr, int shift);

        double CompareShifted(const Description& l, const Description &r, int shift=0, int shiftY = 0);

        double Compare(const Description& l, const Description &r);
    public:

        inline int clamp(int val, int min, int max){
            return std::max(std::min(max, val), min);
        }
    };
}