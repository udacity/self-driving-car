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

#include <NeuralNetwork/Neuron.h>

SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::NeuronInterface::Factory, NeuralNetwork::Neuron, NeuralNetwork::Neuron::deserialize)
SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::NeuronInterface::Factory, NeuralNetwork::BiasNeuron, NeuralNetwork::BiasNeuron::deserialize)
SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::NeuronInterface::Factory, NeuralNetwork::InputNeuron, NeuralNetwork::InputNeuron::deserialize)

SimpleJSON::Type::Object NeuralNetwork::Neuron::serialize() const{
	return {
		{"class", "NeuralNetwork::Neuron"},
		{"id", id},
		{"output", output()},
		{"value", value()},
		{"activationFunction", *activation},
		{"basisFunction", *basis},
		{"weights", weights}
	};
}

std::unique_ptr<NeuralNetwork::Neuron> NeuralNetwork::Neuron::deserialize(const SimpleJSON::Type::Object &obj) {
	Neuron *neuron = new Neuron(obj["id"].as<int>());
	neuron->_output=obj["output"].as<double>();
	neuron->_value=obj["value"].as<double>();
	delete neuron->activation;
	delete neuron->basis;
	neuron->activation=NeuralNetwork::ActivationFunction::Factory::deserialize(obj["activationFunction"].as<SimpleJSON::Type::Object>()).release();
	neuron->basis=NeuralNetwork::BasisFunction::Factory::deserialize(obj["basisFunction"].as<SimpleJSON::Type::Object>()).release();
	const SimpleJSON::Type::Array& weights=obj["weights"].as<SimpleJSON::Type::Array>();
	neuron->weights.resize(weights.size());
	for(std::size_t i=0;i<weights.size();i++) {
		neuron->weights[i]=weights[i].as<double>();
	}
	return std::unique_ptr<NeuralNetwork::Neuron>(neuron);
}


