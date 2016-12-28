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

#include <NeuralNetwork/FeedForward/Network.h>

SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::FeedForward::Network::Factory, NeuralNetwork::FeedForward::Network,NeuralNetwork::FeedForward::Network::deserialize)

std::vector<float> NeuralNetwork::FeedForward::Network::computeOutput(const std::vector<float>& input) {
	// 0 is bias
	_partialInput[0]=1.0;
	for(std::size_t i=0;i<input.size();i++) {
		_partialInput[i+1]=input[i];
	}

	for(std::size_t i=1;i<layers.size();i++) {
		layers[i]->solve(_partialInput,_partialOutput);
		_partialInput.swap(_partialOutput);
	}

	return std::vector<float>(_partialInput.begin()+1,_partialInput.begin()+outputs()+1);
}

void NeuralNetwork::FeedForward::Network::randomizeWeights() {
	for(std::size_t layerIndex=1;layerIndex<layers.size();layerIndex++) {
		auto &layer=layers[layerIndex];
		auto &prevLayer=layers[layerIndex-1];

		for(std::size_t neuron=1; neuron < layer->size(); neuron ++ ) {
			for(std::size_t prevNeuron=0; prevNeuron < prevLayer->size(); prevNeuron++) {
				layer->operator[](neuron).weight(prevNeuron)=1.0-static_cast<float>(rand()%2001)/1000.0;
			}
		}
	}

}

std::unique_ptr<NeuralNetwork::FeedForward::Network> NeuralNetwork::FeedForward::Network::deserialize(const SimpleJSON::Type::Object &obj) {
	NeuralNetwork::FeedForward::Network* network= new NeuralNetwork::FeedForward::Network();
	for(auto layers:network->layers) {
		delete layers;
	}

	network->layers.clear();

	for(auto& layerObject: obj["layers"].as<SimpleJSON::Type::Array>()) {
		network->layers.push_back(NeuralNetwork::FeedForward::Layer::Factory::deserialize(layerObject.as<SimpleJSON::Type::Object>()).release());

		if(network->_partialInput.size() < network->layers.back()->size()) {
			network->_partialInput.resize(network->layers.back()->size());
		}

		if(network->_partialOutput.size() < network->layers.back()->size()) {
			network->_partialOutput.resize(network->layers.back()->size());
		}

	}

	network->_inputs=network->layers[0]->size()-1;
	network->_outputs=network->layers.back()->size()-1;

	return std::unique_ptr<Network>(network);
}

