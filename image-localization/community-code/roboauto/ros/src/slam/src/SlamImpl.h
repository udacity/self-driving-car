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
 * @brief: A core class of this project containing the Localize() method.
 *
 * Performs localization based on the set of our descriptors.
 * For position estimation we used a particle filter and its augmented extension.
 * There are also some other techniques for stabilizing the PF and maintaining a precise position.
 */

#pragma once

#include "WeightFunction.h"

#include "Filter.h"

#include <Motion.h>

#include "TrafficLightDetector.h"
#include "BridgeDetector.h"
#include <utils/Clusterization.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/property_tree/ptree.hpp>

#include <thread>

namespace Slam {
    class SlamImpl {
    public:
        SlamImpl();
        ~SlamImpl();

        void LoadMap(const std::string &filename);
        void SaveMap(const std::string &filename);
    protected:
        /* core method Localize() */
        // <segment position, variance>
        std::pair<utils::GpsCoords,double> Localize();
        void DetectObjects(const cv::Mat &img, const cv::Mat &gray);

        std::size_t PARTICLE_NUMBER = 1500;
        float PARTICLE_NUMBER_AUGMENTED_PERCENTAGE = 0.02;

        /* Position history */
        static constexpr bool USE_LAST_POSITION_AUGMENTED = false;
        uint16_t LAST_POSITION_WINDOW_SIZE = 40;
        uint16_t LAST_POSITION_JUMP_COUNT = 80;
        size_t positionHistoryErrorCounter_ = 0;
        double JUMP_TRESHOLD_COEF;

        size_t positionCounter_ = 0;
        std::vector<double> posHistory_ = {};

        double augmentedHorizontThreshold_ = 0.19;
        double augmentedMinSize_ = 100;
        double augmentedFrom_ = 0;
        double augmentedTo_ = 0;
        double augmentedFromSize_ = 0;
        double augmentedToSize_ = 0;
        double augmentedSizeCoefUp_ = 0.0;
        double augmentedSizeCoefDown_ = 0.0;
        int augmentedCounterLimit_ = 100;
        int augmentedFixCounter_ = 0;

        std::pair<double,double> lastPosition_;
        WeightFunction::Config weightFunctionConfig_;
        ImageDescriptor::Horizont descriptor_;
        TrafficLightDetector trafficLightDetector_;
        utils::PoleDetector poleDetector_;
        BridgeDetector::BridgeDetector bridge_;

        utils::Map::Map map_;

        utils::ParticleFilter::ParticleFilter<1,0> particleFilter_;
        utils::Clusterization clusterization_;
        bool particleFilterInitialized_ = false;
        double odoMetry_ = 0.0;
        struct CurrentState {
            double distance = 0;
            double distanceError = 0;
            utils::DetectedObjects detectedObjects;
            // Boost serialization
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive & ar, const unsigned int version) {
                ar & distance;
                ar & distanceError;
                ar & detectedObjects;
            }
        } state_;

        boost::property_tree::ptree tree_;

        void Reset();
        void PostProcess();
    private:
        double AvgShift() const;
        bool mapProcessed_ = false;
        std::pair<double,double> ComputeNewPosition();
        bool PostProcessLights(const size_t segment,const size_t mapPoint);

        int nrOfInitialResamplingSteps_ = 2;
    };
}