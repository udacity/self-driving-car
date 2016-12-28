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

#include "./Network.h"
#include <NeuralNetwork/ActivationFunction/Heaviside.h>

namespace NeuralNetwork {
namespace FeedForward {
	class Perceptron : private Network{
		public:
			inline Perceptron(size_t _inputSize,size_t _outputSize):Network(_inputSize) {
				appendLayer(_outputSize,ActivationFunction::Heaviside(0.0));
			};

			using Network::computeOutput;
			using Network::randomizeWeights;

			inline std::size_t size() const {
				return layers[1]->size();
			}

			inline NeuronInterface& operator[](const std::size_t& neuron) {
				return layers[1]->operator[](neuron);
			}

		protected:
	};
}
}