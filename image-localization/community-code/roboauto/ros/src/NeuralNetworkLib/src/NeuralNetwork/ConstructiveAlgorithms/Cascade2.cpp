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

#include <NeuralNetwork/ConstructiveAlgorithms/Cascade2.h>

using namespace NeuralNetwork::ConstructiveAlgorihtms;

std::pair<std::shared_ptr<NeuralNetwork::Neuron>, std::vector<float>> Cascade2::trainCandidates(Cascade::Network &network,
																								std::vector<std::shared_ptr<Neuron>> &candidates,
																								const std::vector<TrainingPattern> &patterns) {
	std::size_t outputs = patterns[0].second.size();

	std::vector<TrainingPattern> patternsForOutput;

	float sumSqDiffs=0.0;

	for(auto &pattern:patterns) {
		patternsForOutput.emplace_back(getInnerNeuronsOutput(network, pattern.first), pattern.second);
	}

	std::vector <std::vector<float>> errors(patterns.size());

	for(std::size_t patternNumber = 0; patternNumber < patterns.size(); patternNumber++) {
		auto &pattern = patterns[patternNumber];
		errors[patternNumber].resize(network.outputs());

		std::vector<float> output = network.computeOutput(patterns[patternNumber].first);
		for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
			float diff = output[outputIndex]-pattern.second[outputIndex];
			errors[patternNumber][outputIndex] = diff;
			sumSqDiffs+=diff*diff;
		}
	}

	std::size_t iterations = 0;
	std::size_t iterationsWithoutIprovement = 0;
	float bestCorrelation = 0;
	float lastCorrelation = 0;
	std::size_t bestCandidateIndex=0;
	std::shared_ptr<Neuron> bestCandidate = nullptr;

	std::vector<std::vector<float>> candidateWeights(candidates.size());
	for(auto &w: candidateWeights) {
		w.resize(outputs);
		for(auto &output: w) {
			output = fabs(_distribution(_generator))*0.5;
		}
	}

	//compute Correlation Epoch
	do {
		lastCorrelation = bestCorrelation;
		bool firstStep = true;
		std::size_t candidateIndex=0;

		for(auto &candidate : candidates) {

			float score=sumSqDiffs;
			std::vector<float> slopes(candidate->getWeights().size());
			std::vector<float> outSlopes(outputs);

			std::size_t patternIndex=0;
			for(auto &pattern:patternsForOutput) {
				float errSum = 0.0;
				float activationValue =(*candidate)(pattern.first);
				float derivatived = candidate->getActivationFunction().derivatedOutput(candidate->value(), candidate->output());

				for(std::size_t output = 0; output < outputs; output++) {
					float weight = candidateWeights[candidateIndex][output];
					float diff = activationValue * weight - errors[patternIndex][output];

					score -= (diff * diff);
					outSlopes[output] -= 2.0 * diff * activationValue;
					errSum += diff * weight;
					patternIndex++;
				}
				errSum*= derivatived;

				for(std::size_t input = 0; input < pattern.first.size(); input++) {
					slopes[input] -= errSum*pattern.first[input];
				}
			}

			for(std::size_t weightIndex = 0; weightIndex < slopes.size(); weightIndex++) {
				candidate->weight(weightIndex) += slopes[weightIndex] * 0.7 / (patterns.size());/// (patterns.size() * patterns[0].first.size());
			}

			for(std::size_t weightIndex = 0; weightIndex < outSlopes.size(); weightIndex++) {
				candidateWeights[candidateIndex][weightIndex] += outSlopes[weightIndex] * 0.7  / (patterns.size());/// (patterns.size() * patterns[0].first.size());
			}

			if(firstStep || score > bestCorrelation) {
				bestCorrelation = score;
				bestCandidate = candidate;
				firstStep = false;
				bestCandidateIndex=candidateIndex;
			}
			candidateIndex++;
		}

		if(bestCorrelation <= lastCorrelation) {
			iterationsWithoutIprovement++;
		}

	}
	while(iterations++ < _maxCandidateIterations && iterationsWithoutIprovement < _maxCandidateIterationsWithoutChange);
	std::cout << "iter: " << iterations << ", correlation: " << bestCorrelation << ", " << lastCorrelation << "\n";

	for(auto &a : candidateWeights[bestCandidateIndex]) {
		a*=-1.0;
	}

	return {bestCandidate, candidateWeights[bestCandidateIndex]};
}

/*
 *
std::pair<std::shared_ptr<NeuralNetwork::Neuron>, std::vector<float>> Cascade2::trainCandidates(Cascade::Network &network,
																										  std::vector<std::shared_ptr<Neuron>> &candidates,
																										  const std::vector<TrainingPattern> &patterns) {
	std::size_t outputs = patterns[0].second.size();

	std::vector<TrainingPattern> patternsForOutput;

	float sumSqDiffs=0.0;

	for(auto &pattern:patterns) {
		patternsForOutput.emplace_back(getInnerNeuronsOutput(network, pattern.first), pattern.second);
	}

	std::vector<float> errors(patterns.size());

	for(std::size_t patternNumber = 0; patternNumber < patterns.size(); patternNumber++) {
		auto &pattern = patterns[patternNumber];

		std::vector<float> output = network.computeOutput(patterns[patternNumber].first);
		for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
			float diff = output[outputIndex]-pattern.second[outputIndex];
			errors[outputIndex] += diff;
			sumSqDiffs+=diff*diff;
		}
	}

	std::size_t iterations = 0;
	std::size_t iterationsWithoutIprovement = 0;
	float bestCorrelation = 0;
	float lastCorrelation = 0;
	std::size_t bestCandidateIndex=0;
	std::shared_ptr<Neuron> bestCandidate = nullptr;

	std::vector<std::vector<float>> candidateWeights(candidates.size());
	for(auto &w: candidateWeights) {
		w.resize(outputs);
		for(auto &output: w) {
			output = fabs(_distribution(_generator));
		}
	}

	//compute Correlation Epoch
	do {
		lastCorrelation = bestCorrelation;
		bool firstStep = true;
		std::size_t candidateIndex=0;

		for(auto &candidate : candidates) {

			float score=sumSqDiffs;
			std::vector<float> slopes(candidate->getWeights().size());
			std::vector<float> outSlopes(outputs);

			for(auto &pattern:patternsForOutput) {
				float errSum = 0.0;
				float activationValue =(*candidate)(pattern.first);
				float derivatived = candidate->getActivationFunction().derivatedOutput(candidate->value(), candidate->output());

				for(std::size_t output = 0; output < outputs; output++) {
					float weight = candidateWeights[candidateIndex][output];
					float diff = activationValue * weight - errors[output];

					float goalDir= pattern.second[output] <0.0? -1.0 :1.0;
					float diffDir= diff >0.0? -1.0 :1.0;
					score -= (diff * diff);
					outSlopes[output] += diff * activationValue;
					errSum += diff * weight;
				}
				errSum*= derivatived;

				for(std::size_t input = 0; input < pattern.first.size(); input++) {
					slopes[input] += errSum*pattern.first[input];
				}
			}

			for(std::size_t weightIndex = 0; weightIndex < slopes.size(); weightIndex++) {
				candidate->weight(weightIndex) += slopes[weightIndex] * 0.7/ (patterns.size() * patterns[0].first.size());
			}

			for(std::size_t weightIndex = 0; weightIndex < outSlopes.size(); weightIndex++) {
				candidateWeights[candidateIndex][weightIndex] += outSlopes[weightIndex] * 0.7/ (patterns.size() * patterns[0].first.size());
			}

			if(firstStep || score > bestCorrelation) {
				bestCorrelation = score;
				bestCandidate = candidate;
				firstStep = false;
				bestCandidateIndex=candidateIndex;
			}
			candidateIndex++;
		}

		if(bestCorrelation <= lastCorrelation) {
			iterationsWithoutIprovement++;
		}

	}
	while(iterations++ < _maxCandidateIterations && iterationsWithoutIprovement < _maxCandidateIterationsWithoutChange);
	std::cout << "iter: " << iterations << ", correlation: " << bestCorrelation << ", " << lastCorrelation << "\n";

	for(auto &a : candidateWeights[bestCandidateIndex]) {
		a*=-1.0;
	}

	return {bestCandidate, candidateWeights[bestCandidateIndex]};
}
*/


/*
std::pair<std::shared_ptr<NeuralNetwork::Neuron>, std::vector<float>> Cascade2::trainCandidates(Cascade::Network &network,
																								std::vector<std::shared_ptr<Neuron>> &candidates,
																								const std::vector<TrainingPattern> &patterns) {
	std::size_t outputs = patterns[0].second.size();

	std::vector<TrainingPattern> patternsForOutput;
	std::vector<FeedForward::Network*> patternNets;

	for(auto &pattern:patterns) {
		patternsForOutput.emplace_back(getInnerNeuronsOutput(network, pattern.first), pattern.second);
	}

	std::vector<float> errors(patterns.size());

	for(std::size_t patternNumber = 0; patternNumber < patterns.size(); patternNumber++) {
		auto &pattern = patterns[patternNumber];

		std::vector<float> output = network.computeOutput(patterns[patternNumber].first);
		patternNets.push_back(new FeedForward::Network(patternsForOutput[patternNumber].first.size()-1));
		auto patternNetwork = patternNets.back();

		auto &hidden = patternNetwork->appendLayer(2);
		auto &outputLayer = patternNetwork->appendLayer(outputs);

		for(std::size_t outputIndex = 0; outputIndex < outputs; outputIndex++) {
			outputLayer[outputIndex+1].weight(0) = network.getOutputNeurons()[outputIndex]->value();
			float diff = pattern.second[outputIndex] - output[outputIndex];
			errors[outputIndex] += diff;
		}
	}

	std::size_t iterations = 0;
	std::size_t iterationsWithoutIprovement = 0;
	float bestCorrelation = 0;
	float lastCorrelation = 0;
	std::size_t bestCandidateIndex=0;

	std::vector<std::vector<float>> candidateWeights(candidates.size());

	for(auto &w: candidateWeights) {
		w.resize(outputs);
		for(auto &output: w) {
			output = fabs(_distribution(_generator));
		}
	}
	std::vector<float>candidateScores(candidates.size());
	//compute Correlation Epoch
	do {
		std::fill(candidateScores.begin(),candidateScores.end(),0.0);
		lastCorrelation = bestCorrelation;

		for(std::size_t patternIndex=0;patternIndex<patternsForOutput.size();patternIndex++) {
			std::size_t candidateIndex=0;
			auto &pattern = patternsForOutput[patternIndex];
			auto net = patternNets[patternIndex];
			Learning::BackPropagation bp(*net);

			for(auto &candidate : candidates) {
				float score = 0;
				(*net)[1][1].setWeights(candidate->getWeights());

				for(std::size_t outputNeuron=0;outputNeuron<outputs;outputNeuron++) {
					(*net)[2][outputNeuron+1].weight(1)=candidateWeights[candidateIndex][outputNeuron];
				}

				bp.teach(pattern.first,pattern.second);
				auto res = net->computeOutput(pattern.first);

				for(std::size_t outputNeuron=0;outputNeuron<outputs;outputNeuron++) {
					candidateWeights[candidateIndex][outputNeuron]=(*net)[2][outputNeuron+1].weight(1);
					candidateScores[candidateIndex]+=res[outputNeuron]*res[outputNeuron];
				}
				candidate->setWeights((*net)[1][1].getWeights());
				candidateIndex++;
			}

		}

		bestCorrelation=candidateScores[0];
		bestCandidateIndex=0;

		for(std::size_t index=1;index < candidateScores.size();index++) {
			if(bestCorrelation > candidateScores[index]) {
				bestCandidateIndex = index;
			}
		}

		if(bestCorrelation <= lastCorrelation) {
			iterationsWithoutIprovement++;
		}

	}
	while(iterations++ < _maxCandidateIterations && iterationsWithoutIprovement < _maxCandidateIterationsWithoutChange);
	std::cout << "iter: " << iterations << ", correlation: " << bestCorrelation << ", " << lastCorrelation << "\n";

	for(auto &net:patternNets) {
		delete net;
	}
	return {candidates[bestCandidateIndex], candidateWeights[bestCandidateIndex]};
}
*/