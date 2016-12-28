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

#include <NeuralNetwork/Recurrent/Network.h>
#include <iostream>

SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::Recurrent::Network::Factory, NeuralNetwork::Recurrent::Network, NeuralNetwork::Recurrent::Network::deserialize)

std::vector<float> NeuralNetwork::Recurrent::Network::computeOutput(const std::vector<float>& input, unsigned int iterations) {

	assert(input.size() == _inputs);

	if(_outputsOfNeurons.size() != neurons.size()) {
		_outputsOfNeurons.resize(neurons.size());
		for(auto &neuron:neurons) {
			_outputsOfNeurons[neuron->id]=neuron->output();
		}
	}

	std::vector<float> newOutputs(neurons.size());

	for(size_t i=0;i<_inputs;i++) {
		_outputsOfNeurons[i+1]=input[i];
		newOutputs[i+1]=input[i];
	}

	newOutputs[0]=neurons[0]->output();

	std::size_t neuronsSize = neurons.size();

	for(unsigned int iter=0;iter< iterations;iter++) {
		for(size_t i=_inputs+1;i<neuronsSize;i++) {
			newOutputs[i] = neurons[i]->operator()(_outputsOfNeurons);
		}
		_outputsOfNeurons.swap(newOutputs);
	}

	std::vector<float> ret;
	for(size_t i=0;i<_outputs;i++) {
		ret.push_back(neurons[i+_inputs+1]->output());
	}

	return ret;
}

NeuralNetwork::Recurrent::Network NeuralNetwork::Recurrent::Network::connectWith(const NeuralNetwork::Recurrent::Network &) const {

}

NeuralNetwork::Recurrent::Network& NeuralNetwork::Recurrent::Network::operator=(const NeuralNetwork::Recurrent::Network&r) {
	NeuralNetwork::Network::operator=(r);

	for(std::size_t i=1;i<neurons.size();i++) {
		delete neurons[i];
	}

	neurons.resize(1);

	for(std::size_t i=1;i<neurons.size();i++) {
		neurons.push_back(r.neurons[i]->clone());
	}
	return *this;
}

SimpleJSON::Type::Object NeuralNetwork::Recurrent::Network::serialize() const {
	std::vector<SimpleJSON::Value> neuronsSerialized;
	for(auto &neuron: neurons) {
		neuronsSerialized.push_back(neuron->serialize());
	}
	return {
		{"class", "NeuralNetwork::Recurrent::Network"},
		{"inputSize", _inputs},
		{"outputSize", _outputs},
		{"outputs", _outputsOfNeurons},
		{"neurons", neuronsSerialized}
	};
}

std::unique_ptr<NeuralNetwork::Recurrent::Network> NeuralNetwork::Recurrent::Network::deserialize(const SimpleJSON::Type::Object &obj) {
	const int inputSize=obj["inputSize"].as<int>();
	const int outputSize=obj["outputSize"].as<int>();
	NeuralNetwork::Recurrent::Network *net= new NeuralNetwork::Recurrent::Network(inputSize,outputSize,0) ;
	for(auto &a: net->neurons) {
		delete a;
	}
	net->neurons.clear();
	for(const auto& neuronObj: obj["neurons"].as<SimpleJSON::Type::Array>()) {
		NeuronInterface* neuron=Neuron::Neuron::Factory::deserialize(neuronObj.as<SimpleJSON::Type::Object>()).release();
		net->neurons.push_back(neuron);
	}
	return std::unique_ptr<Network>(net);
}

/*
NeuralNetwork::Recurrent::Network NeuralNetwork::Recurrent::Network::connectWith(const NeuralNetwork::Recurrent::Network &r) const {
	if(outputSize!=r.inputSize) {
		//TODO: throw exception
	}

	NeuralNetwork::Recurrent::Network newNetwork(inputSize,r.outputSize,(neurons.size()-1-inputSize)+(r.neurons.size()-1-r.inputSize-r.outputSize));

	// update output neurons first
	for(size_t i=0;i<r.outputSize;i++) {
		size_t index=1+newNetwork.inputSize+i;

		delete newNetwork.neurons[index];
		newNetwork.neurons[index]= r.neurons[1+r.inputSize+i]->clone();

		Neuron* n= newNetwork.neurons[index];

		for(int i=0;i<newNetwork.inputSize;i++) {
			n->setWeight(newNetwork.+i,0.0);
		}

		for(int i=0;i<r.inputSize;i++) {
			n->setWeight(1+newNetwork.inputSize+newNetwork.outputSize,0.0);
		}
	}

	return newNetwork;
}
*/