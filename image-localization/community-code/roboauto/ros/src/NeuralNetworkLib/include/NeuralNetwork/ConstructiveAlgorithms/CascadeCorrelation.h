/*
 *  Copyright 2016 Tomas Cernik, Tom.Cernik@gmail.com
 *  All rights reserved.
 *
 *  This file is part of NeuralNetworkLib
 *
 *  NeuralNetworkLib is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  NeuralNetworkLib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NeuralNetworkLib.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../Cascade/Network.h"
#include "../FeedForward/Network.h"
#include "../Learning/QuickPropagation.h"

#include <random>
#include <algorithm>

namespace NeuralNetwork {
	namespace ConstructiveAlgorihtms {
		class CascadeCorrelation {
			public:
				typedef std::pair<std::vector<float>, std::vector<float>> TrainingPattern;

				CascadeCorrelation(std::size_t numberOfCandidate = 18, float maxError = 0.7) :
					_errorTreshold(maxError), _weightRange(0.3), _numberOfCandidates(numberOfCandidate), _generator(rand()), _distribution() {
					setWeightRange(_weightRange);
				}

				virtual ~CascadeCorrelation() {

				}

				virtual Cascade::Network construct(const std::vector<TrainingPattern> &patterns) {
					std::size_t inputs = patterns[0].first.size();
					std::size_t outputs = patterns[0].second.size();

					Cascade::Network network(inputs, outputs, *_activFunction.get());

					network.randomizeWeights();

					_epoch = 0;
					_neurons = 0;
					float error;
					float lastError;
					if(_maxRandomOutputWeights) {
						error = trainOutputsRandom(0, network, patterns);
					} else {
						error = trainOutputs(network, patterns);
					}

					std::cout << error << "\n";

					while(_epoch++ < _maxEpochs && _neurons < _maxHiddenUnits && error > _errorTreshold) {
						std::vector<std::shared_ptr<Neuron>> candidates = createCandidates(network.getNeuronSize() - outputs);

						std::pair<std::shared_ptr<Neuron>, std::vector<float>> candidate = trainCandidates(network, candidates, patterns);

						addBestCandidate(network, candidate);

						lastError=error;
						if(_maxRandomOutputWeights) {
							error = trainOutputsRandom(_epoch, network, patterns);
						} else {
							error = trainOutputs(network, patterns);
						}

						std::cout << error << "\n";

						if(_pruningStatus && error >= lastError * _pruningLimit) { // it is not getting bettter
							network.removeLastHiddenNeuron();
							error=lastError;
							std::cout << "PRUNED\n";
						} else {
							_neurons++;
						}
					}

					return network;
				}

				float getWeightRange() const {
					return _weightRange;
				}

				void setWeightRange(float weightRange) {
					_weightRange = weightRange;
					_distribution = std::uniform_real_distribution<>(-weightRange, weightRange);
				}

				void setMaximumHiddenNeurons(std::size_t neurons) {
					_maxHiddenUnits = neurons;
				}

				void setMaximumEpochs(std::size_t epochs) {
					_maxEpochs = epochs;
				}

				void setActivationFunction(const ActivationFunction::ActivationFunction &function) {
					_activFunction = std::shared_ptr<ActivationFunction::ActivationFunction>(function.clone());
				}

				void setProbabilisticOutputWeightSearch(std::size_t number) {
					_maxRandomOutputWeights = number;
				}

				std::size_t getProbabilisticOutputWeightSearch() const {
					return _maxRandomOutputWeights;
				}

				std::size_t getEpochs() const {
					return _epoch;
				}

				void setPruningLimit(float limit) {
					_pruningStatus=true;
					_pruningLimit=limit;
				}

				bool getPruningLimit() const {
					return _pruningLimit;
				}

				void setErrorThreshold(float err) {
					_errorTreshold = err;
				}

				std::size_t getErrorThreshold() const {
					return _errorTreshold;
				}

				void setMaxCandidateIterationsWithoutChange(std::size_t iter) {
					_maxCandidateIterationsWithoutChange = iter;
				}

				std::size_t getMaxCandidateIterationsWithoutChange() const {
					return _maxCandidateIterationsWithoutChange;
				}

				void setMaxCandidateIterations(std::size_t iter) {
					_maxCandidateIterations = iter;
				}

				std::size_t getMaxCandidateIterations() const {
					return _maxCandidateIterations;
				}

				void setMaxOutpuLearningIterationsWithoutChange(std::size_t iter) {
					_maxOutputLearningIterationsWithoutChange = iter;
				}

				std::size_t getMaxOutpuLearningIterationsWithoutChange() const {
					return _maxOutputLearningIterationsWithoutChange;
				}

				void setMaxOutpuLearningIterations(std::size_t iter) {
					_maxOutputLearningIterations = iter;
				}

				std::size_t getMaxOutpuLearningIterations() const {
					return _maxOutputLearningIterations;
				}

				std::size_t getNumberOfCandidates() const {
					return _numberOfCandidates;
				}

				void setNumberOfCandidates(std::size_t numberOfCandidates) {
					_numberOfCandidates = numberOfCandidates;
				}

			protected:
				std::shared_ptr<ActivationFunction::ActivationFunction> _activFunction = std::make_shared<ActivationFunction::Sigmoid>(-1.0);
				float _minimalErrorStep = 0.00005;
				float _errorTreshold;
				float _weightRange;

				bool _pruningStatus = false;
				float _pruningLimit=0.0;

				std::size_t _epoch = 0;
				std::size_t _neurons = 0;
				std::size_t _maxEpochs = 20;
				std::size_t _maxHiddenUnits = 20;
				std::size_t _maxRandomOutputWeights = 0;
				std::size_t _numberOfCandidates;

				std::size_t _maxOutputLearningIterations = 1000;
				std::size_t _maxOutputLearningIterationsWithoutChange = 5;
				std::size_t _maxCandidateIterations = 4000;
				std::size_t _maxCandidateIterationsWithoutChange = 5;

				std::mt19937 _generator;
				std::uniform_real_distribution<> _distribution;

				std::vector<float> getInnerNeuronsOutput(Cascade::Network &network, const std::vector<float> &input) {
					std::vector<float> output = network.computeOutput(input);
					std::vector<float> outputOfUnits(network.getNeuronSize() - output.size());
					outputOfUnits[0] = 1.0;
					for(std::size_t i = 0; i < input.size(); i++) {
						outputOfUnits[i + 1] = input[i];
					}

					for(std::size_t i = input.size() + 1; i < network.getNeuronSize() - output.size(); i++) {
						outputOfUnits[i] = network.getNeuron(i)->output();
					}
					return outputOfUnits;
				}

				virtual float trainOutputs(Cascade::Network &network, const std::vector<TrainingPattern> &patterns);

				virtual float trainOutputsRandom(std::size_t step, Cascade::Network &network, const std::vector<TrainingPattern> &patterns);

				virtual std::pair<std::shared_ptr<Neuron>, std::vector<float>> trainCandidates(Cascade::Network &network, std::vector<std::shared_ptr<Neuron>> &candidates,
																					   const std::vector<TrainingPattern> &patterns);

				void addBestCandidate(Cascade::Network &network, const std::pair<std::shared_ptr<Neuron>, std::vector<float>> &candidate) {
					auto neuron = network.addNeuron();

					float weightPortion = network.getNeuronSize() - network.outputs();
					neuron->setWeights(candidate.first->getWeights());
					neuron->setActivationFunction(candidate.first->getActivationFunction());
					std::size_t outIndex = 0;
					for(auto &n :network.getOutputNeurons()) {
						auto weights = n->getWeights();
						for(auto &weight: weights) {
							weight *= 0.9;
						}
						n->setWeights(weights);
						n->weight(n->getWeights().size() - 1) = candidate.second[outIndex] / weightPortion;
						outIndex++;
					}
				}

				std::vector<std::shared_ptr<Neuron>> createCandidates(std::size_t id) {
					std::vector<std::shared_ptr<Neuron>> candidates;

					for(std::size_t i = 0; i < _numberOfCandidates; i++) {
						candidates.push_back(std::make_shared<Neuron>(id));
						candidates.back()->setInputSize(id);
						candidates.back()->setActivationFunction(*_activFunction.get());

						for(std::size_t weightIndex = 0; weightIndex < id; weightIndex++) {
							candidates.back()->weight(weightIndex) = _distribution(_generator);// * 3.0;
						}
					}
					return candidates;
				}
		};

	}
}