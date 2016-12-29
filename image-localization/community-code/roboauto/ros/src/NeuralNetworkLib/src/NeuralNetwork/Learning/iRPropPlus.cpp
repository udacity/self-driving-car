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

#include <NeuralNetwork/Learning/iRPropPlus.h>

void NeuralNetwork::Learning::iRPropPlus::updateWeightsAndEndBatch() {
	float error = 0.0;
	const auto& outputLayer=_network[_network.size()-1];
	for(std::size_t j=1;j<outputLayer.size();j++) {
		error+=_slopes[_network.size()-1][j];
	}

	error /= outputLayer.size();

	for(std::size_t layerIndex=1;layerIndex<_network.size();layerIndex++) {
		auto &layer = _network[layerIndex];
		auto &prevLayer = _network[layerIndex - 1];

		std::size_t prevLayerSize = prevLayer.size();
		std::size_t layerSize = layer.size();

		for(std::size_t j = 1; j < layerSize; j++) {
			for(std::size_t k = 0; k < prevLayerSize; k++) {
				float gradient = _gradients[layerIndex][j][k];
				float lastGradient = _lastGradients[layerIndex][j][k];

				_lastGradients[layerIndex][j][k] = gradient;

				float weightChangeDelta = _changesOfWeightChanges[layerIndex][j][k];
				float delta;

				if(gradient * lastGradient > 0) {
					weightChangeDelta = std::min(weightChangeDelta*weightChangePlus,maxChangeOfWeights);
					delta = (std::signbit(gradient)? 1.0f : -1.0f ) * weightChangeDelta;
					layer[j].weight(k) -= delta;
				} else if (gradient * lastGradient < 0) {
					weightChangeDelta = std::max(weightChangeDelta*weightChangeMinus,minChangeOfWeights);
					delta = _lastWeightChanges[layerIndex][j][k];
					if(error > _prevError) {
						layer[j].weight(k) += delta;
					}
					_lastGradients[layerIndex][j][k] = 0;
				} else {
					delta = (std::signbit(gradient)? 1.0f : -1.0f ) * weightChangeDelta;
					layer[j].weight(k) -= delta;
				}
				//std::cout << delta <<"\n";

				_changesOfWeightChanges[layerIndex][j][k] = weightChangeDelta;
				_lastWeightChanges[layerIndex][j][k] = delta;
			}
		}
	}
	_prevError = error;
}