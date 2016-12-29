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

#include "SlamImpl.h"

#include <utils/SerializationHelper.h>
#include <future>
#include <ros/package.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/serialization/vector.hpp>
#include <ImageDescriptor/Horizont.h>

Slam::SlamImpl::SlamImpl() :
    descriptor_(),
    clusterization_(map_)

{
    namespace pt = boost::property_tree;

    static const auto configFilename = ros::package::getPath("slam") + "/config.json";
    pt::read_json(configFilename, tree_);

    std::cout << "====  INIT  =======\n";

    nrOfInitialResamplingSteps_ = tree_.get<int>("init.nrOfInitialResamplingSteps");
    std::cout << "Number of resampling steps after init: " << nrOfInitialResamplingSteps_ << std::endl;

    LAST_POSITION_WINDOW_SIZE = tree_.get<int>("jumpDetection.positionWindowSize");
    LAST_POSITION_JUMP_COUNT = tree_.get<int>("jumpDetection.jumpCount");
    JUMP_TRESHOLD_COEF  = tree_.get<double>("jumpDetection.jumpTreshCoef");

    std::cout << "\n====  WF  =======\n";

    weightFunctionConfig_.horizontParam = tree_.get<double>("weightFunction.horizontParam");
    weightFunctionConfig_.lightsPositive = tree_.get<double>("weightFunction.lightsPositive");
    weightFunctionConfig_.lightsNegative = tree_.get<double>("weightFunction.lightsNegative");
    weightFunctionConfig_.polesPositive = tree_.get<double>("weightFunction.polesPositive");
    weightFunctionConfig_.polesNegative = tree_.get<double>("weightFunction.polesNegative");
    weightFunctionConfig_.signsPositive = tree_.get<double>("weightFunction.signsPositive");
    weightFunctionConfig_.signsNegative = tree_.get<double>("weightFunction.signsNegative");

    std::cout << "Horizont param: " << weightFunctionConfig_.horizontParam << std::endl;
    std::cout << "Lights pos: " << weightFunctionConfig_.lightsPositive << " neg: " << weightFunctionConfig_.lightsNegative << std::endl;
    std::cout << "Signs pos: " << weightFunctionConfig_.signsPositive << " neg: " << weightFunctionConfig_.signsNegative << std::endl;

    std::cout << "\n==== PF =====\n";
    PARTICLE_NUMBER = tree_.get<int>("particleFilter.particles");
    PARTICLE_NUMBER_AUGMENTED_PERCENTAGE = tree_.get<double>("particleFilter.augmentedCoeficient");
    std::cout << "Particles size: " << PARTICLE_NUMBER << std::endl;
    std::cout << "Augmented particles size coeficient: " << PARTICLE_NUMBER_AUGMENTED_PERCENTAGE << std::endl;

    std::cout << "\n==== AUGMENTED =====\n";
    augmentedHorizontThreshold_ = tree_.get<double>("augmented.horizontThreshold");
    augmentedSizeCoefUp_ = tree_.get<double>("augmented.sizeCoefUp");
    augmentedSizeCoefDown_ = tree_.get<double>("augmented.sizeCoefDown");
    augmentedMinSize_ = tree_.get<int>("augmented.minAugmentedSize");
    augmentedCounterLimit_ =tree_.get<int>("augmented.counterLimit");

    std::cout << "Max jump: " << augmentedHorizontThreshold_ << " m" << std::endl;
    std::cout << "Augmented size coef up: " << augmentedSizeCoefUp_ << std::endl;
    std::cout << "Augmented size coef down: " << augmentedSizeCoefDown_ << std::endl;
    std::cout << "min augmented size: " << augmentedMinSize_ << " m" << std::endl;

    std::cout << "\n==== CLUSTERIZATION =====\n";
    clusterization_.SURROUNDING_METERS = tree_.get<double>("clusterization.surroundungMeters");
    std::cout << "Surrounding meters: " << clusterization_.SURROUNDING_METERS << " m" << std::endl;

    descriptor_.config_.rRange = tree_.get<int>("horizontRoughness.range");
    descriptor_.config_.rMaxJump = tree_.get<int>("horizontRoughness.maxJump");
    descriptor_.config_.rCoef = tree_.get<double>("horizontRoughness.coef");
    std::cout << std::endl;

    int seed = tree_.get<int>("seed");
    if(seed == 0) {
        seed = time(NULL);
        std::cout << "Using seed " << seed << std::endl;
    }
    srand(seed);
}

Slam::SlamImpl::~SlamImpl()
{
}

/**
 * The localize method wraps initialization, particle filter steps and the computation of a new position.
 * @return
 */
std::pair<utils::GpsCoords,double> Slam::SlamImpl::Localize() {
    auto fun =std::make_shared<WeightFunction>(map_, descriptor_,state_.detectedObjects,weightFunctionConfig_);

    particleFilter_.SetWeightFunction(fun);
    std::pair<double,double> pos;
    if(!particleFilterInitialized_) {
        particleFilter_.EnableThreads();
        particleFilter_.SetParticleNumber(PARTICLE_NUMBER);
        particleFilter_.Clear();
        particleFilter_.GeneratePopulation(InitializationFunction(static_cast<int>(0),map_.GetSegmentsSize()-1,PARTICLE_NUMBER));
        particleFilterInitialized_=true;
        // let the PF converge from initial frame
        for(int i=0; i<nrOfInitialResamplingSteps_; i++)
            particleFilter_.Step(ResamplingFunction(map_,0.1,0.1));
        pos = ComputeNewPosition();
        particleFilter_.SetParticleNumber(PARTICLE_NUMBER);
    } else {
        odoMetry_ += fabs(state_.distance);
        // PF step every x meters (not resampling while the car is not moving)
        if(odoMetry_ > 0.15) {
            // Adding some particles for augmented mode (adding some random particles)
            particleFilter_.AddPopulation(InitializationFunction(static_cast<int>(augmentedFrom_),static_cast<int>(augmentedTo_)),
                                          static_cast<size_t>(PARTICLE_NUMBER * PARTICLE_NUMBER_AUGMENTED_PERCENTAGE));
            particleFilter_.Step(ResamplingFunction(map_, odoMetry_, state_.distanceError));
            pos = ComputeNewPosition();
            odoMetry_ = 0.0;
        }else {
            pos = lastPosition_;
        }
    }

    lastPosition_ = pos;

    double variance = pos.second;
    double bestPositionInSegment = pos.first;
    utils::GpsCoords coords = map_.GetSegment(static_cast<size_t>(floor(bestPositionInSegment))).GetBeginning();
    coords.Shift(map_.GetSegment(static_cast<size_t>(floor(bestPositionInSegment))).GetDir()*(bestPositionInSegment-floor(bestPositionInSegment)));

    return {coords,variance};
}

/**
 * This method computes a new position based on the particle distribution across the map.
 * The method also adjusts the boundaries of the augmented mode. The adjustments are based on the best position horizon score.
 * That roughly translates to the boundaries of the augmented mode moving away while the score is below threshold.
 * On the other hadn, while the score is high enough, the boundaries are getting closer and thus reduce the range in which the particles are generated.
 *
 * @return <position, variance> (the integer part denotes a map segment, the part after floating point denotes a percentage position in the segment)
 */
std::pair<double,double> Slam::SlamImpl::ComputeNewPosition() {
    clusterization_.Update(particleFilter_.Particles());
    auto pos = clusterization_.GetBestCluster();

    auto fun =WeightFunction(map_, descriptor_,state_.detectedObjects,weightFunctionConfig_);
    utils::ParticleFilter::ParticleFilter<1, 0>::Particle entity;
    entity[0] = pos.first;
    auto bestParticleWeight = fun(entity);

    auto prevPos = lastPosition_.first;

    // update augmented mode boundaries
    if (bestParticleWeight <= augmentedHorizontThreshold_) {
        augmentedFixCounter_--;
        double coef = augmentedFixCounter_ > 0 ? augmentedSizeCoefUp_ : pow(augmentedSizeCoefUp_, sqrt(abs(augmentedFixCounter_)));

        if(pos.first > prevPos) {
            augmentedToSize_ = std::max(augmentedMinSize_, augmentedToSize_ * coef);
            augmentedFromSize_ = std::max(augmentedMinSize_, std::max(augmentedFromSize_, fabs(pos.first - augmentedFrom_)) * coef);
        } else {
            augmentedToSize_ = std::max(augmentedMinSize_, std::max(augmentedToSize_, fabs(pos.first - augmentedTo_)) * coef );
            augmentedFromSize_ = std::max(augmentedMinSize_, augmentedFromSize_ * coef);
        }

        augmentedFrom_ = pos.first - augmentedFromSize_;
        augmentedTo_ = pos.first + augmentedToSize_;

    } else {
        augmentedFixCounter_++;
        double coef = augmentedFixCounter_ < 0 ? augmentedSizeCoefDown_ : pow(augmentedSizeCoefDown_, sqrt(augmentedFixCounter_));
        augmentedToSize_ = std::max(augmentedMinSize_, augmentedToSize_ / coef);
        augmentedFromSize_ = std::max(augmentedMinSize_, augmentedFromSize_ / coef);

        augmentedFrom_ = pos.first - augmentedFromSize_;
        augmentedTo_ = pos.first + augmentedToSize_;
    }

    augmentedFixCounter_ = std::max(std::min(augmentedFixCounter_, augmentedCounterLimit_), -augmentedCounterLimit_);

    augmentedTo_ = std::min(std::max(augmentedTo_, 0.0), static_cast<double>(map_.GetSegmentsSize() - 1));
    augmentedFrom_ = std::min(std::max(augmentedFrom_, 0.0), static_cast<double>(map_.GetSegmentsSize() - 1));


    if (USE_LAST_POSITION_AUGMENTED) {

        double avgShift = AvgShift();
        double lastPos = lastPosition_.first;

        double distance = map_.GetDistBetweenSegments(lastPos,pos.first); // distance

        double compShift = 0;
        uint8_t count = 1; // intentionally 1 and not 0, not a mistake (hopefully)
        for (auto it = posHistory_.begin(); it < posHistory_.end()-1; ++it) {
            ++count;
            compShift += map_.GetDistBetweenSegments(*std::next(it), *it);
        }
        compShift /= count;

        if(positionCounter_ < LAST_POSITION_WINDOW_SIZE) {
            std::rotate(posHistory_.begin(), posHistory_.begin() + 1, posHistory_.end());
            posHistory_[LAST_POSITION_WINDOW_SIZE-1] = pos.first;
            positionCounter_++;
        }else if(distance > compShift*JUMP_TRESHOLD_COEF) {
            ++positionHistoryErrorCounter_;
            if(positionHistoryErrorCounter_ > LAST_POSITION_JUMP_COUNT) {
                positionHistoryErrorCounter_ = 0;
            } else {
                pos.first = posHistory_[LAST_POSITION_WINDOW_SIZE-1] + avgShift;
                //pos.first =map_.ShiftByMeters(posHistory_[LAST_POSITION_WINDOW_SIZE-1],avgShift);
            }
        } else {
            positionHistoryErrorCounter_ = 0;
        }

        std::rotate(posHistory_.begin(), posHistory_.begin() + 1, posHistory_.end());
        posHistory_[LAST_POSITION_WINDOW_SIZE-1] = pos.first;
    }

    return pos;
};

double Slam::SlamImpl::AvgShift() const
{
    double compShift = 0;
    uint8_t count = 1; // intentionally 1 and not 0, not a mistake (hopefully)
    for (auto it = posHistory_.begin(); it < posHistory_.end()-1; ++it) {
        ++count;
        compShift += fabs(*std::next(it) - *it);
        //compShift += map_.GetDistBetweenSegments(*std::next(it), *it);//fabs(*std::next(it) - *it);
    }
    return compShift / count;
}

void Slam::SlamImpl::SaveMap(const std::string &filename) {
    std::ofstream ofs(filename);
    boost::archive::binary_oarchive oa(ofs);

    oa << map_;
}
void Slam::SlamImpl::LoadMap(const std::string &filename) {
    std::ifstream ifs(filename);

    if (!ifs.good()) {
        std::cout << "file does not exist" << std::endl;
        return;
    }

    boost::archive::binary_iarchive ia(ifs);
    ia >> map_;
    mapProcessed_ = false;
    PostProcess();
}

/**
 * Performs all the detections in a given image.
 * @param img
 * @param gray
 */
void Slam::SlamImpl::DetectObjects(const cv::Mat &img, const cv::Mat &gray) {

    auto detection1 = std::async(std::launch::async, [this,img]() {
        // Object detection
        state_.detectedObjects.lights = trafficLightDetector_.FindTrafficLights(img);
        state_.detectedObjects.horizont = descriptor_.DescribeImage(img);
    });

    auto detection2 = std::async(std::launch::async, [this,img,gray]() {
        state_.detectedObjects.poles = poleDetector_.FindPolesInImage(gray);
        state_.detectedObjects.bridge = bridge_.findBridge(img);

        cv::Mat hsv;
        cv::cvtColor(img, hsv, CV_BGR2HSV);

        state_.detectedObjects.yellowSigns = utils::SignDetection::DetectYellowMarks(hsv);
        state_.detectedObjects.darkYellowSigns = utils::SignDetection::DetectDarkYellowMarks(hsv);
        state_.detectedObjects.darkGreenSigns = utils::SignDetection::DetectDarkGreen(hsv);
    });

    detection1.get();
    detection2.get();
}

/**
 * Computes some values that are not saved in the map file.
 */
void Slam::SlamImpl::PostProcess() {
    if(!mapProcessed_) {
        for (size_t i = 0; i < map_.GetSegmentsSize(); ++i) {
            for (utils::Map::MapPoint &mp : map_.GetSegment(i).GetMapPoints()) {
                mp.detectedObjecsts.horizont.maximums = ImageDescriptor::Horizont::detectMax(mp.detectedObjecsts.horizont.horizont);
            }

//            for (size_t j = 0; j < map_.GetSegment(i).GetMapPoints().size(); ++j) {
//                map_.GetSegment(i).GetMapPoints()[j].detectedObjecsts.semaphore = PostProcessLights(i,j);
//            }
        }
        mapProcessed_ = true;
    }
}

bool Slam::SlamImpl::PostProcessLights(const size_t segment,const size_t mapPoint)
{
    if (segment <= 1 || segment >= map_.GetSegmentsSize() -2) return false;

    const int sigma = 10;
    size_t cnt = 0;

    int curSegment = segment;
    size_t curMapPoint = mapPoint;

    //move to first map point
    int remSigma = sigma;
    while (remSigma > 0) {
        remSigma -= curMapPoint;

        if (remSigma > 0) {
            --curSegment;
            curMapPoint = map_.GetSegment(curSegment).GetMapPoints().size() - 1;
        }
    }

    if (remSigma > 0) curMapPoint -= remSigma;

    for (int i = -sigma; i < sigma; ++i) {
        if (!map_.GetSegment(curSegment).GetMapPoints()[curMapPoint].detectedObjecsts.lights.empty()) {
            cnt++;
        }

        ++curMapPoint;
        if (curMapPoint == map_.GetSegment(curSegment).GetMapPoints().size()) {
            curMapPoint = 0;
            ++curSegment;
        }
    }

    return cnt > sigma*2/3;
}

void Slam::SlamImpl::Reset() {
    odoMetry_ = 0.0;
    particleFilterInitialized_ = false;
    augmentedFromSize_ = map_.GetSegmentsSize()/2.0;
    augmentedToSize_ = map_.GetSegmentsSize()/2.0;
    augmentedFrom_ = 0;
    augmentedTo_ = map_.GetSegmentsSize()-1;
    augmentedFixCounter_ = 0;
    posHistory_.resize(LAST_POSITION_WINDOW_SIZE);
    std::fill(posHistory_.begin(), posHistory_.end(),0);
    positionHistoryErrorCounter_ = 0;
    positionCounter_ = 0;
    clusterization_.Reset();
}