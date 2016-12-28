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

#include "../Network.h"
#include <random>

namespace NeuralNetwork {
	namespace Cascade {
		class Network : public NeuralNetwork::Network {
			public:
				/**
				 * @brief Constructor for Network
				 * @param _inputSize is number of inputs to network
				 */
				Network(std::size_t inputSize, std::size_t outputSize, const ActivationFunction::ActivationFunction &activationFunction=ActivationFunction::Sigmoid(-4.9)) : NeuralNetwork::Network(inputSize,outputSize) {
					_neurons.push_back(std::make_shared<BiasNeuron>());

					for(std::size_t i = 0; i < inputSize; i++) {
						_neurons.push_back(std::make_shared<InputNeuron>(_neurons.size()));
					}

					for(std::size_t i = 0; i < outputSize; i++) {
						_neurons.push_back(std::make_shared<Neuron>(_neurons.size(),activationFunction));
						_neurons.back()->setInputSize(inputSize + 1); // +1 is bias
					}
				}

				virtual std::vector<float> computeOutput(const std::vector<float> &input) override {
					std::vector<float> compute;
					compute.resize(_neurons.size());

					compute[0] = 1.0;

					for(std::size_t i = 0; i < _inputs; i++) {
						compute[i+1] = input[i];
					}

					// 0 is bias, 1-_inputSize is input
					for(std::size_t i = _inputs + 1; i < _neurons.size(); i++) {
						compute[i] = (*_neurons[i].get())(compute);
					}

					return std::vector<float>(compute.end() - _outputs, compute.end());
				}

				std::size_t getNeuronSize() const {
					return _neurons.size();
				}

				const std::vector<std::shared_ptr<NeuronInterface>>& getNeurons() {
					return _neurons;
				}

				std::shared_ptr<NeuronInterface> getNeuron(std::size_t id) {
					return _neurons[id];
				}

				std::vector<std::shared_ptr<NeuronInterface>> getOutputNeurons() {
					return std::vector<std::shared_ptr<NeuronInterface>>(_neurons.end()-_outputs,_neurons.end());
				}

				void removeLastHiddenNeuron() {
					_neurons.erase(_neurons.begin()+_neurons.size()-outputs()-1);

					std::size_t maxIndexOfHiddenNeuron = _neurons.size() - outputs();

					std::size_t maxIndexOfNeuron = _neurons.size() - 1;

					for(std::size_t i = 0; i < _outputs; i++) {
						_neurons[maxIndexOfNeuron-i]->setInputSize(maxIndexOfHiddenNeuron);
					}
				}

				std::shared_ptr<NeuronInterface> addNeuron() {
					_neurons.push_back(std::make_shared<Neuron>());
					auto neuron = _neurons.back();
					neuron->setInputSize(_neurons.size() - _outputs-1);
					// 0 is bias, 1-_inputSize is input
					std::size_t maxIndexOfNeuron = _neurons.size() - 1;
					// move output to right position
					for(std::size_t i = 0; i < _outputs; i++) {
						std::swap(_neurons[maxIndexOfNeuron - i], _neurons[maxIndexOfNeuron - i - 1]);
					}

					for(std::size_t i = 0; i < _outputs; i++) {
						_neurons[maxIndexOfNeuron - i]->setInputSize(_neurons.size() - _outputs);
					}
					return neuron;
				}

				virtual SimpleJSON::Type::Object serialize() const override {
					std::vector<SimpleJSON::Value> neuronsSerialized;
					for(auto &neuron: _neurons) {
						neuronsSerialized.push_back(neuron->serialize());
					}

					return {
						{"class",      "NeuralNetwork::Recurrent::Network"},
						{"inputSize",  _inputs},
						{"outputSize", _outputs},
						{"neurons",    neuronsSerialized}
					};
				}

				static std::unique_ptr<Network> deserialize(const SimpleJSON::Type::Object &obj) {
					const int inputSize = obj["inputSize"].as<int>();
					const int outputSize = obj["outputSize"].as<int>();
					Network *net = new Network(inputSize, outputSize);
					net->_neurons.clear();

					for(const auto &neuronObj: obj["neurons"].as<SimpleJSON::Type::Array>()) {
						net->_neurons.push_back(Neuron::Factory::deserialize(neuronObj.as<SimpleJSON::Type::Object>()));
					}

					return std::unique_ptr<Network>(net);
				}

				//I I H H O O 6
				void randomizeWeights() {
					std::mt19937 _generator(rand());
					std::uniform_real_distribution<> _distribution(-0.3,0.3);
					for(auto& neuron :getOutputNeurons()) {
						for(std::size_t weight = 0; weight < neuron->getWeights().size(); weight++) {
							neuron->weight(weight) = _distribution(_generator);
						}
					}
				}

			protected:
				std::vector<std::shared_ptr<NeuronInterface>> _neurons = {};

			SIMPLEJSON_REGISTER(NeuralNetwork::Cascade::Network::Factory, NeuralNetwork::Cascade::Network, deserialize)
		};
	}
}