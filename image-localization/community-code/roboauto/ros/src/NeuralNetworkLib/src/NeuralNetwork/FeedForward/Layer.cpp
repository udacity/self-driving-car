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

#include <NeuralNetwork/FeedForward/Layer.h>

SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::FeedForward::Layer::Factory, NeuralNetwork::FeedForward::Layer,NeuralNetwork::FeedForward::Layer::deserialize)

void NeuralNetwork::FeedForward::Layer::solve(const std::vector<float> &input, std::vector<float> &output) {
	if(output.size() < neurons.size()) {
		output.resize(neurons.size());
	}

	for(auto&neuron: neurons) {
		output[neuron->id] = neuron->operator()(input);
	}
}

SimpleJSON::Type::Object NeuralNetwork::FeedForward::Layer::serialize() const {
	std::vector<SimpleJSON::Value> neuronsSerialized;
	for(std::size_t i=0;i<neurons.size();i++) {
		neuronsSerialized.push_back(neurons[i]->serialize());
	}
	return {{"class", "NeuralNetwork::FeedForward::Layer"},
			{"neurons" , neuronsSerialized}
	};
}

std::unique_ptr<NeuralNetwork::FeedForward::Layer> NeuralNetwork::FeedForward::Layer::deserialize(const SimpleJSON::Type::Object &obj) {
	NeuralNetwork::FeedForward::Layer *layer= new NeuralNetwork::FeedForward::Layer();
	for(auto& neuron: obj["neurons"].as<SimpleJSON::Type::Array>()) {
		layer->neurons.push_back(NeuralNetwork::NeuronInterface::Factory::deserialize(neuron.as<SimpleJSON::Type::Object>()).release());
	}
	return std::unique_ptr<Layer>(layer);
}


