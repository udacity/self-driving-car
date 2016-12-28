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

#include <NeuralNetwork/ConstructiveAlgorithms/CascadeCorrelation.h>

#include <NeuralNetwork/Learning/BackPropagation.h>

using namespace NeuralNetwork::ConstructiveAlgorihtms;

float CascadeCorrelation::trainOutputs(Cascade::Network &network, const std::vector <CascadeCorrelation::TrainingPattern> &patterns) {
	std::size_t outputs = patterns[0].second.size();

	FeedForward::Network p(network.getNeuronSize() - outputs - 1);
	p.appendLayer(outputs);
	Learning::BackPropagation learner(p);

	for(std::size_t neuron = 0; neuron < outputs; neuron++) {
		p[1][neuron + 1].setWeights(network.getOutputNeurons()[neuron]->getWeights());
		p[1][neuron + 1].setActivationFunction(network.getOutputNeurons()[neuron]->getActivationFunction());
	}

	std::vector <TrainingPattern> patternsForOutput;

	for(auto &pattern:patterns) {
		patternsForOutput.emplace_back(getInnerNeuronsOutput(network, pattern.first), pattern.second);
	}

	float lastError;
	float error = std::numeric_limits<float>::max();
	std::size_t iteration = 0;
	std::size_t iterWithoutImporvement = 0;
	do {
		lastError = error;
		for(auto &pattern:patternsForOutput) {
			learner.teach({pattern.first.begin() + 1, pattern.first.end()}, pattern.second);
		}

		error = 0;
		for(auto &pattern:patternsForOutput) {
			std::vector<float> output = p.computeOutput({pattern.first.begin() + 1, pattern.first.end()});
			for(std::size_t outputIndex = 0; outputIndex < output.size(); outputIndex++) {
				error += pow(output[outputIndex] - pattern.second[outputIndex], 2);
			}
		}

		error/=patterns.size();

		if(fabs(lastError - error) < _minimalErrorStep) {
			iterWithoutImporvement++;
		} else {
			iterWithoutImporvement = 0;
		}
	}
	while(iteration++ < _maxOutputLearningIterations && iterWithoutImporvement < _maxOutputLearningIterationsWithoutChange);

//	std::cout << "outputLearning: " << error << ", last: " << lastError << ", iters: " << iteration << "\n";

	for(std::size_t neuron = 0; neuron < outputs; neuron++) {
		network.getOutputNeurons()[neuron]->setWeights(p[1][neuron + 1].getWeights());
	}
	return error;
}


float CascadeCorrelation::trainOutputsRandom(std::size_t step, Cascade::Network &network, const std::vector <CascadeCorrelation::TrainingPattern> &patterns) {
	std::size_t outputs = patterns[0].second.size();

	std::vector < FeedForward::Network * > possibleOutputs;
	{ // first networks is special
		possibleOutputs.emplace_back(new FeedForward::Network(network.getNeuronSize() - outputs - 1));
		FeedForward::Network &p = (*possibleOutputs.back());
		p.appendLayer(outputs);

		for(std::size_t neuron = 0; neuron < outputs; neuron++) {
			p[1][neuron + 1].setWeights(network.getNeuron(network.getNeuronSize() - outputs + neuron)->getWeights());
			p[1][neuron + 1].setActivationFunction(network.getNeuron(network.getNeuronSize() - outputs + neuron)->getActivationFunction());
		}

	}

	std::size_t generatedNets = 0;

	if(step == 0) {
		generatedNets = _maxRandomOutputWeights;
	} else if(step % 15 == 0) {
		generatedNets = _maxRandomOutputWeights;
	} else {
		generatedNets = _maxRandomOutputWeights / step;
	}

	for(std::size_t net = 0; net < generatedNets; net++) {
		possibleOutputs.emplace_back(new FeedForward::Network(network.getNeuronSize() - outputs - 1));
		FeedForward::Network &p = (*possibleOutputs.back());
		p.appendLayer(outputs);
		for(std::size_t neuron = 0; neuron < outputs; neuron++) {
			for(std::size_t weight = 0; weight < network.getNeuronSize() - outputs - 1; weight++) {
				p[1][neuron + 1].weight(weight) = _distribution(_generator);
			}
			p[1][neuron + 1].setActivationFunction(network.getNeuron(network.getNeuronSize() - outputs + neuron)->getActivationFunction());
		}
	}

	std::vector <TrainingPattern> patternsForOutput;

	for(auto &pattern:patterns) {
		patternsForOutput.emplace_back(getInnerNeuronsOutput(network, pattern.first), pattern.second);
	}

	std::size_t bestNetwork = 0;
	float bestScore = std::numeric_limits<float>::max();
	std::size_t index = 0;

	for(auto &net : possibleOutputs) {
		auto &p = *net;
		Learning::BackPropagation learner(p);

		float lastError;
		float error = std::numeric_limits<float>::max();
		std::size_t iteration = 0;
		std::size_t iterWithoutImporvement = 0;
		do {
			lastError = error;
			for(auto &pattern:patternsForOutput) {
				learner.teach({pattern.first.begin() + 1, pattern.first.end()}, pattern.second);
			}

			error = 0;
			for(auto &pattern:patternsForOutput) {
				std::vector<float> output = p.computeOutput({pattern.first.begin() + 1, pattern.first.end()});
				for(std::size_t outputIndex = 0; outputIndex < output.size(); outputIndex++) {
					error += pow(output[outputIndex] - pattern.second[outputIndex], 2);
				}
			}

			error/=patterns.size();

			if(fabs(lastError - error) < _minimalErrorStep) {
				iterWithoutImporvement++;
			} else {
				iterWithoutImporvement = 0;
			}
		}
		while(iteration++ < _maxOutputLearningIterations && iterWithoutImporvement < _maxOutputLearningIterationsWithoutChange);
		if(error < bestScore) {
			bestScore = error;
			bestNetwork = index;
		}
		index++;
	}

	FeedForward::Network &p = *possibleOutputs[bestNetwork];

	std::cout << "network: " << bestNetwork << "\n";

	for(std::size_t neuron = 0; neuron < outputs; neuron++) {
		network.getNeuron(network.getNeuronSize() - outputs + neuron)->setWeights(p[1][neuron + 1].getWeights());
	}
	return bestScore;
}

std::pair <std::shared_ptr<NeuralNetwork::Neuron>, std::vector<float>> CascadeCorrelation::trainCandidates(Cascade::Network &network,
																										   std::vector <std::shared_ptr<Neuron>> &candidates,
																										   const std::vector <TrainingPattern> &patterns) {
	std::size_t outputs = patterns[0].second.size();

	std::vector <TrainingPattern> patternsForOutput;

	for(auto &pattern:patterns) {
		patternsForOutput.emplace_back(getInnerNeuronsOutput(network, pattern.first), pattern.second);
	}

	std::vector <std::vector<float>> errors(patterns.size());
	std::vector<float> meanErrors(outputs);
	float sumSquareError = 0;

	std::vector <std::vector<float>> errorsReal(patterns.size());
	for(std::size_t patternNumber = 0; patternNumber < patterns.size(); patternNumber++) {
		auto &pattern = patterns[patternNumber];
		errors[patternNumber].resize(network.outputs());
		errorsReal[patternNumber].resize(network.outputs());

		std::vector<float> output = network.computeOutput(patterns[patternNumber].first);
		for(std::size_t outputIndex = 0; outputIndex < network.outputs(); outputIndex++) {
			float diff = output[outputIndex] - pattern.second[outputIndex];
			//float diff = pattern.second[outputIndex] - output[outputIndex];

			auto neuron = network.getOutputNeurons()[outputIndex];
			float derivation = neuron->getActivationFunction().derivatedOutput(neuron->value(), neuron->output());
			float error = derivation * diff;

			errors[patternNumber][outputIndex] = error;
			errorsReal[patternNumber][outputIndex] = error;
			meanErrors[outputIndex] += error;
			sumSquareError += error * error;
		}
	}

	if(sumSquareError < 0.01) {
		sumSquareError=0.01;
	}

	std::for_each(meanErrors.begin(), meanErrors.end(), [&patterns](float &n) { n /= patterns.size(); });

	struct CAND {
		std::vector<float> correlations = {};
		std::vector<float> lastCorrelations = {};
		std::vector<float> slopes = {};
		float sumVals = 0;
		std::shared_ptr <Neuron> candidate = nullptr;
	};

	std::vector <CAND> candidatesRegister(candidates.size());

	for(std::size_t i = 0; i < candidates.size(); i++) {
		candidatesRegister[i].candidate = candidates[i];
		candidatesRegister[i].correlations.resize(outputs);
		candidatesRegister[i].lastCorrelations.resize(outputs);
		candidatesRegister[i].slopes.resize(patternsForOutput[0].first.size());
	}

	std::size_t iterations = 0;
	std::size_t iterationsWithoutIprovement = 0;
	float bestCorrelation = 0;
	float lastCorrelation = 0;
	std::shared_ptr <Neuron> bestCandidate = nullptr;

	std::vector<float> bestCorrelations(outputs);

	for(std::size_t patternIndex = 0; patternIndex < patterns.size(); patternIndex++) {
		for(auto &candidateStruct : candidatesRegister) {
			float value = (*candidateStruct.candidate)(patternsForOutput[patternIndex].first);
			candidateStruct.sumVals += value;
			for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
				candidateStruct.correlations[outputIndex] -= value * meanErrors[outputIndex];
			}
		}
	}

	for(auto &candidate : candidatesRegister) {
		float score = 0.0;
		float aveValue = candidate.sumVals / patterns.size();
		for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
			float correlation = (candidate.correlations[outputIndex] - aveValue * meanErrors[outputIndex]) / sumSquareError;
			candidate.lastCorrelations[outputIndex] = correlation;
			candidate.correlations[outputIndex] = 0;
			candidate.sumVals = 0;
			score += fabs(correlation);
		}
	}

	do {
		lastCorrelation = bestCorrelation;

		/*compute correlations */
		for(std::size_t patternIndex = 0; patternIndex < patterns.size(); patternIndex++) {
			for(auto &candidateStruct : candidatesRegister) {
				auto candidate = candidateStruct.candidate;
				float change = 0;
				float activation = (*candidate)(patternsForOutput[patternIndex].first);
				candidateStruct.sumVals += activation;

				float derivation = candidate->getActivationFunction().derivatedOutput(candidate->value(), candidate->output()) / sumSquareError;
				for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
					float error = errors[patternIndex][outputIndex];

					float direction = candidateStruct.lastCorrelations[outputIndex] < 0.0 ? -1.0 : 1.0;

					change -= direction * derivation * (error - meanErrors[outputIndex]);

					candidateStruct.correlations[outputIndex] -= error * activation;
				}

				for(std::size_t i = 0; i < candidateStruct.slopes.size(); i++) {
					candidateStruct.slopes[i] += change * patternsForOutput[patternIndex].first[i];
				}
			}
		}

		/*Update Weights*/
		for(auto &candidateStruct : candidatesRegister) {
			auto candidate = candidateStruct.candidate;
			for(std::size_t i = 0; i < candidateStruct.slopes.size(); i++) {
				candidate->weight(i) += candidateStruct.slopes[i];
				candidateStruct.slopes[i] = 0.0;
			}
		}

		/* adjust correlations*/
		bestCorrelation = 0;
		bool step = true;
		for(auto &candidate : candidatesRegister) {
			float score = 0.0;
			float aveValue = candidate.sumVals / patterns.size();
			for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
				float correlation = (candidate.correlations[outputIndex] - aveValue * meanErrors[outputIndex]) / sumSquareError;
				candidate.lastCorrelations[outputIndex] = correlation;
				candidate.correlations[outputIndex] = 0;
				candidate.sumVals = 0;
				score += fabs(correlation);
			}

			if(score > bestCorrelation || step) {
				bestCandidate = candidate.candidate;
				bestCorrelation = score;
				bestCorrelations = candidate.lastCorrelations;
				step = false;
			}
		}

		if(bestCorrelation <= lastCorrelation) {
			iterationsWithoutIprovement++;
		}
//		std::cout << "sub iter: " << iterations << ", correlation: " << bestCorrelation << ", " << lastCorrelation << "\n";
	}
	while(iterations++ < _maxCandidateIterations && iterationsWithoutIprovement < _maxCandidateIterationsWithoutChange);
	std::cout << "iter: " << iterations << ", correlation: " << bestCorrelation << ", " << lastCorrelation << "\n";

	for(auto&a: bestCorrelations) {
		a*=-1.0;
	}

	return {bestCandidate, bestCorrelations};
}