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
 * @brief: Particle filter implementation along with some base classes for helper classes that are utilized in the particle filter.
 */

#pragma once

#include <vector>
#include "./Random.h"
#include "General.h"

#include <opencv2/core/utility.hpp>

#include <ros/console.h>

namespace utils {
    namespace ParticleFilter {

        class Exception : public std::exception {
            public:
                Exception(const std::string& what) : what_(what) {

                }

                const char* what() const noexcept {
                    return what_.c_str();
                }
            private:
                std::string what_;
        };

        template<unsigned int DIM, unsigned int ANGLES = 1>
        class ParticleFilter {
        public:
            static const std::size_t PERCENTAGE_OF_INLIERS = 10;

            class Particle {
            public:
                static const unsigned int dimensions = DIM;
                static const unsigned int angles = ANGLES;

                Particle() : weight_(FMATH_NAN)
                {

                }

                const double& operator[](const std::size_t& i) const
                {
                    return position_[i];
                }

                double& operator[](const std::size_t& i)
                {
                    return position_[i];
                }

                void SetWeight(const double& weight)
                {
                    weight_ = weight;
                }

                const double& Weight() const
                {
                    return weight_;
                }

            protected:
                double position_[DIM] = {0};
                double weight_;
            };

            class WeightFunction {
            public:
                virtual double operator()(const Particle& entity) = 0;
            };

            class ResamplingFunction {
            public:
                virtual void Resample(Particle& entity) const
                {
                    for (std::size_t i = 0; i < Particle::dimensions; i++) {
                        entity[i] += 0.5 - static_cast<double>(rand() % 100) / 100.0;
                    }
                };

                virtual void Normalize(Particle& entity) const
                {

                }
            };

            class InitializationFunction {
            public:
                virtual void initialize(Particle& entity) const
                {
                    Random::Normal random;
                    for (std::size_t i = 0; i < Particle::dimensions; i++) {
                        entity[i] = random();
                    }
                }
            };

            ParticleFilter(const std::size_t& particleNumber = 500) : weightFunction_(), particleNumber_(particleNumber)
            {

            }

            void EnableThreads() {
                enableThreads_=true;
            }

            void DisableThreads() {
                enableThreads_=false;
            }

            // generates populatuion with set distribution
            void GeneratePopulation(const InitializationFunction& initMethod = InitializationFunction());

            void AddPopulation(const InitializationFunction& initMethod = InitializationFunction(), std::size_t numOfParticles = 100);

            void AddParticle(const Particle& particle) {
                particles_.push_back(particle);
            }

            void Clear() {
                particles_.clear();
            }

            // Step of filtering with set resampling model_
            void Step(const ResamplingFunction& resamplingModel = ResamplingFunction());

            // sets weight function
            void SetWeightFunction(std::shared_ptr<WeightFunction> method)
            {
                weightFunction_ = method;
            }

            // gets number of particles
            std::size_t GetParticleNumber()
            {
                return particleNumber_;
            }

            // sets number of particles
            void SetParticleNumber(const std::size_t& weightNumber)
            {
                particleNumber_ = weightNumber;
            }

            // returns vector of particles in current step
            std::vector<Particle>& Particles()
            {
                return particles_;
            }

            // returns vector of particles in current step
            const std::vector<Particle>& Particles() const
            {
                return particles_;
            }

            // returns best particle in current step
            static const Particle GetMeanParticle(const std::vector<Particle> &particles ,std::pair<Particle,std::vector<double>> filter={})
            {
                Particle mean;

                std::size_t particlesNotFilteredOut=0;
                std::vector<bool> particleFilteredOut;

                if(filter.second.size()) {
                    particleFilteredOut.resize(particles.size());
                    for (std::size_t i = 0; i < particles.size(); i++) {
                        bool filterOut = false;

                        for (std::size_t dim = 0; !filterOut && dim < DIM-ANGLES; dim++) {
                            if((particles[i][dim]-filter.first[dim]) > filter.second[dim]) {
                                filterOut=true;
                            }
                        }

                        for (std::size_t dim = DIM-ANGLES; dim < DIM; dim++) {
                            if((particles[i][dim]-filter.first[dim]) > filter.second[dim]) {
                                filterOut=true;
                            }
                        }

                        particleFilteredOut[i]=filterOut;
                        particlesNotFilteredOut+=!filterOut;
                    }
                }
                else
                {
                    particlesNotFilteredOut = particles.size();
                }

                // compute mean for dimensions without angles
                for (std::size_t i = 0; i < particles.size(); i++) {
                    if(!filter.second.size() || !particleFilteredOut[i]) {
                        for (std::size_t dim = 0; dim < DIM-ANGLES; dim++) {
                            mean[dim] += particles[i][dim];
                        }
                    }
                }

                for (std::size_t dim = 0; dim < DIM-ANGLES; dim++) {
                    mean[dim] /= particlesNotFilteredOut;
                }

                // compute mean for angles according to https://en.wikipedia.org/wiki/Mean_of_circular_quantities
                for (std::size_t dim = DIM-ANGLES; dim < DIM; dim++) {
                    double sinSum=0.0;
                    double cosSum=0.0;
                    for (std::size_t i = 0; i < particles.size(); i++) {
                        if(!filter.second.size() || !particleFilteredOut[i]) {
                            sinSum += sin(particles[i][dim]);
                            cosSum += cos(particles[i][dim]);
                        }
                    }
                    mean[dim]=std::atan2(sinSum,cosSum);
                }
                if(particlesNotFilteredOut == 0) {
                    throw Exception("All Particles filtered out");
                }

                if(filter.second.size()) {
                    ROS_DEBUG_STREAM(particlesNotFilteredOut << "/" << particles.size());
                }

                return mean;
            }

            std::vector<double> computeVariance(const Particle& mean) const {
                std::vector<double> variance(DIM);

                /// compute variance for dimensions without angles
                for (std::size_t i = 0; i < particles_.size(); i++) {
                    for (std::size_t dim = 0; dim < DIM-ANGLES; dim++) {
                        variance[dim] += pow(particles_[i][dim] - mean[dim], 2);
                    }
                }

                for (std::size_t dim = 0; dim < DIM-ANGLES; dim++) {
                    variance[dim] = variance[dim] / particles_.size();
                }

                /// compute variance for angles
                for (std::size_t dim = DIM-ANGLES; dim < DIM; dim++) {
                    double sinSum=0.0;
                    double cosSum=0.0;
                    for (std::size_t i = 0; i < particles_.size(); i++) {
                        sinSum+=sin(particles_[i][dim]);
                        cosSum+=cos(particles_[i][dim]);
                    }

                    // according to http://www.ebi.ac.uk/thornton-srv/software/PROCHECK/nmr_manual/man_cv.html
                    variance[dim] = 1.0-sqrt(pow(sinSum, 2) + pow(cosSum, 2))/particles_.size();
                }

                return variance;
            }

            // returns best particle in current step removes outliers using JackKnife-like method
            const Particle GetMeanWithoutOutliersParticle()
            {
                Particle mean = GetMeanParticle(particles_);

                std::vector<double> sqVariance = computeVariance(mean); //square-rooted variance

                // compute square root of variance for dimensions without angles
                for (std::size_t dim = 0; dim < DIM-ANGLES; dim++) {
                    sqVariance[dim] = sqrt(sqVariance[dim] / particles_.size());
                }

                // compute square root of variance for angles
                for (std::size_t dim = DIM-ANGLES; dim < DIM; dim++) {
                    // according to http://www.ebi.ac.uk/thornton-srv/software/PROCHECK/nmr_manual/man_cv.html
                    sqVariance[dim] =sqrt (sqVariance[dim]);
                }

                try {
                    // compute new mean within range (mean,mean+-sqVariance)
                    Particle meanWithoutOutliers = GetMeanParticle(particles_,{mean,sqVariance});
                    return meanWithoutOutliers;
                } catch (const std::exception &except) {
                    return mean;
                }
            }

            void SetResamplePercentage(float value) {
                resamplePercentage = value;
            }

            std::shared_ptr<WeightFunction> weightFunction_;
        protected:

            void ComputeWeights();
            float resamplePercentage = 1.0;
            bool enableThreads_ = false;
            std::size_t iterationNumber_ = 0;
            std::size_t particleNumber_;
            std::vector<Particle> particles_;
        };
    }

    template<unsigned int DIM, unsigned int ANGLES>
    void ParticleFilter::ParticleFilter<DIM,ANGLES>::GeneratePopulation(const InitializationFunction& initMethod)
    {
        particles_.clear();
        iterationNumber_ = 0;
        for (std::size_t counter = 0; counter < particleNumber_; counter++) {
            Particle e;
            initMethod.initialize(e);
            particles_.push_back(e);
        }
    }

    template<unsigned int DIM, unsigned int ANGLES>
    void ParticleFilter::ParticleFilter<DIM,ANGLES>::ComputeWeights()
    {
        if(enableThreads_) {
            class ParallelProcess : public cv::ParallelLoopBody
            {
            public:
                ParallelProcess (ParticleFilter<DIM,ANGLES>& pf) : pf_(pf) {

                }

                void operator() (const cv::Range& range) const override
                {
                    int begin = 0;
                    int end = pf_.Particles().size();
                    if(range != cv::Range::all()) {
                        begin =range.start;
                        end= range.end;
                    }
                    for(int i =begin;i<end;i++) {
                        if (utils::isfnan(pf_.Particles()[i].Weight())) {
                            pf_.Particles()[i].SetWeight(pf_.weightFunction_->operator()(pf_.Particles()[i]));
                        }
                    }
                }
            protected:
                ParticleFilter<DIM,ANGLES>& pf_;
            };

            cv::parallel_for_(cv::Range(0, particles_.size()), ParallelProcess(*this));
        } else {
            for (std::size_t i = 0; i < particles_.size(); i++) {
                if (utils::isfnan(particles_[i].Weight())) {
                    particles_[i].SetWeight(weightFunction_->operator()(particles_[i]));
                }
            }
        }
    }

    template<unsigned int DIM, unsigned int ANGLES>
    void ParticleFilter::ParticleFilter<DIM,ANGLES>::AddPopulation(const InitializationFunction& initMethod, std::size_t numOfParticles)
    {
        for (std::size_t counter = 0; counter < numOfParticles; counter++) {
            Particle e;
            initMethod.initialize(e);
            particles_.push_back(e);
        }
    }

    template<unsigned int DIM, unsigned int ANGLES>
    void ParticleFilter::ParticleFilter<DIM,ANGLES>::Step(const ResamplingFunction& resamplingModel)
    {
        if (iterationNumber_ != 0) {
            for (std::size_t i = 0; i < particles_.size(); i++) {
                resamplingModel.Resample(particles_[i]);
                resamplingModel.Normalize(particles_[i]);

                particles_[i].SetWeight(FMATH_NAN);
            }
        }

        ComputeWeights();

        double weightSum = 0.0;

        std::vector<double> weightPartialSum;

        std::vector<std::size_t> index (particles_.size());
        for(std::size_t i=0;i<particles_.size();i++ ) {
            index[i] = i;
        }

        std::sort(index.begin(),index.end(),[this](std::size_t l, std::size_t r) { return particles_.at(l).Weight() > particles_.at(r).Weight(); });

        std::size_t particleSize = std::min(static_cast<std::size_t>(particles_.size()*resamplePercentage), particles_.size()) ;

        weightPartialSum.resize(particleSize);

        for (std::size_t i = 0; i < particleSize; i++) {
            weightSum += particles_[index[i]].Weight();
            weightPartialSum[i] = weightSum;
        }

        std::vector<Particle> newParticles;
        newParticles.reserve(particleNumber_);

        Random::Uniform random(0.0, weightSum);
        while (newParticles.size() < particleNumber_) {
            double r = random();

            auto it = std::lower_bound(weightPartialSum.begin(), weightPartialSum.end(), r);

            if (it != weightPartialSum.end()) {
                newParticles.push_back(particles_[index[it - weightPartialSum.begin()]]);
            }

            newParticles.back().SetWeight(FMATH_NAN);
        }

        std::swap(newParticles, particles_);
        iterationNumber_++;
    }
}