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

#include <NeuralNetwork/Learning/QuickPropagation.h>

void NeuralNetwork::Learning::QuickPropagation::updateWeightsAndEndBatch() {

	float shrinkFactor=_maxChange/(_maxChange+1.0f);

	for(std::size_t layerIndex=1;layerIndex<_network.size();layerIndex++) {
		auto &layer = _network[layerIndex];
		auto &prevLayer = _network[layerIndex - 1];

		std::size_t prevLayerSize = prevLayer.size();
		std::size_t layerSize = layer.size();

		for(std::size_t j = 1; j < layerSize; j++) {
			for(std::size_t k = 0; k < prevLayerSize; k++) {
				float newChange = 0.0f;
				float _epsilon = 0.9;
				if(fabs (_gradients[layerIndex][j][k])> 0.0001) {
					if(std::signbit(_gradients[layerIndex][j][k]) == std::signbit(_lastGradients[layerIndex][j][k])) {
						newChange+= _gradients[layerIndex][j][k]*_epsilon;

						if(fabs(_gradients[layerIndex][j][k]) > fabs(shrinkFactor * _lastGradients[layerIndex][j][k])) {
							newChange += _maxChange * _gradients[layerIndex][j][k];
						}else {
							newChange+=_gradients[layerIndex][j][k]/(_lastGradients[layerIndex][j][k]-_gradients[layerIndex][j][k]) * _lastDeltas[layerIndex][j][k];
						}
					} else {
						newChange+=_gradients[layerIndex][j][k]/(_lastGradients[layerIndex][j][k]-_gradients[layerIndex][j][k]) * _lastDeltas[layerIndex][j][k];
					}
				} else {
					newChange+= _lastGradients[layerIndex][j][k]*_epsilon;
				}
				_lastDeltas[layerIndex][j][k]= newChange;
				layer[j].weight(k)+= newChange;

	/*          This is according to paper?
	//			delta = _gradients[layerIndex][j][k] / (_lastGradients[layerIndex][j][k]-_gradients[layerIndex][j][k]) * _lastDeltas[layerIndex][j][k];
	//			delta = std::min(_maxChange,delta);
				_lastDeltas[layerIndex][j][k] = delta;
				layer[j].weight(k)+= delta;
	 */
			}
		}
	}
	_lastGradients.swap(_gradients);
}