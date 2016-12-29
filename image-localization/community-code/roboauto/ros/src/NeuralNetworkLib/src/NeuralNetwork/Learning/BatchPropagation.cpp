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

#include <NeuralNetwork/Learning/BatchPropagation.h>

void NeuralNetwork::Learning::BatchPropagation::teach(const std::vector<float> &input, const std::vector<float> &expectation) {
	_network.computeOutput(input);
	if(!init) {
		resize();
		init = true;
	}

	computeSlopes(expectation);

	computeDeltas(input);
	if(++_currentBatchSize >= _batchSize) {
		finishTeaching();
	}
}

void NeuralNetwork::Learning::BatchPropagation::finishTeaching() {
	updateWeightsAndEndBatch();
	_currentBatchSize=0;
}

void NeuralNetwork::Learning::BatchPropagation::computeSlopes(const std::vector<float> &expectation) {
	const auto& outputLayer=_network[_network.size()-1];
	for(std::size_t j=1;j<outputLayer.size();j++) {
		const auto& neuron = outputLayer[j];
		_slopes[_network.size()-1][j]=_correctionFunction->operator()( expectation[j-1], neuron.output())*
									neuron.getActivationFunction().derivatedOutput(neuron.value(),neuron.output());
	}

	for(int layerIndex=static_cast<int>(_network.size()-2);layerIndex>0;layerIndex--) {
		auto &layer=_network[layerIndex];

		for(std::size_t j=1;j<layer.size();j++) {
			float deltasWeight = 0;

			for(std::size_t k=1;k<_network[layerIndex+1].size();k++) {
				deltasWeight+=_slopes[layerIndex+1][k]* _network[layerIndex+1][k].weight(j);
			}

			_slopes[layerIndex][j]=deltasWeight*layer[j].getActivationFunction().derivatedOutput(layer[j].value(),layer[j].output());
		}
	}
}

void NeuralNetwork::Learning::BatchPropagation::computeDeltas(const std::vector<float> &input) {
	for(std::size_t layerIndex=1;layerIndex<_network.size();layerIndex++) {
		auto &layer=_network[layerIndex];
		auto &prevLayer=_network[layerIndex-1];

		std::size_t prevLayerSize=prevLayer.size();
		std::size_t layerSize=layer.size();

		for(std::size_t j=1;j<layerSize;j++)  {
			float update = _slopes[layerIndex][j];
			for(std::size_t k=0;k<prevLayerSize;k++) {
				float inputValue = 0.0;
				if(layerIndex==1 && k!=0) {
					inputValue = input[k-1];
				} else {
					inputValue= prevLayer[k].output();
				}
				if(_currentBatchSize == 0) {
					_gradients[layerIndex][j][k] = update * inputValue;
				} else {
					_gradients[layerIndex][j][k] += update * inputValue;
				}
			}
		}
	}
}

void NeuralNetwork::Learning::BatchPropagation::resize() {
	_slopes.resize(_network.size());

	for(std::size_t i=0; i < _network.size(); i++) {
		_slopes[i].resize(_network[i].size());
	}

	_gradients.resize(_network.size());

	for(std::size_t i = 0; i < _network.size(); i++) {
		_gradients[i].resize(_network[i].size());
		if(i > 0) {
			for(std::size_t j = 0; j < _gradients[i].size(); j++) {
				_gradients[i][j].resize(_network[i - 1].size());
				std::fill(_gradients[i][j].begin(), _gradients[i][j].end(), 0.0);
			}
		}
	}

}
