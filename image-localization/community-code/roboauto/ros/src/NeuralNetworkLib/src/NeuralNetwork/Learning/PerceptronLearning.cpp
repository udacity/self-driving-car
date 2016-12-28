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

#include <NeuralNetwork/Learning/PerceptronLearning.h>

void NeuralNetwork::Learning::PerceptronLearning::teach(const std::vector<float> &input, const std::vector<float> &output) {
	std::vector<float> computedOutput=perceptron.computeOutput(input);

	std::size_t outputSize = output.size();

	for(std::size_t i=0; i<outputSize; i++) {
		perceptron[i+1].weight(0)+=learningCoefficient*(output[i]-computedOutput[i])*1;
		for(std::size_t inputIndex=0; inputIndex<input.size(); inputIndex++) {
			float delta = learningCoefficient*(output[i]-computedOutput[i])*2*(input[inputIndex]-0.5);
			perceptron[i+1].weight(inputIndex+1)+=delta;
		}
	}
}