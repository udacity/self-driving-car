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

#include <vector>

#include <sstream>
#include <iomanip>
#include <limits>

namespace NeuralNetwork {
	namespace Recurrent {

		/**
		* @author Tomas Cernik (Tom.Cernik@gmail.com)
		* @brief Reccurent model of Artifical neural network
		*/
		class Network : public NeuralNetwork::Network {
			public:

				/**
				 * @brief Constructor for Network
				 * @param _inputSize is number of inputs to network
				 * @param _outputSize is size of output from network
				 * @param hiddenUnits is number of hiddenUnits to be created
				 */
				inline Network(size_t inputSize, size_t outputSize, size_t hiddenUnits = 0) : NeuralNetwork::Network(inputSize, outputSize), neurons(0), _outputsOfNeurons(0) {
					neurons.push_back(new NeuralNetwork::BiasNeuron());

					for(size_t i = 0; i < inputSize; i++) {
						neurons.push_back(new NeuralNetwork::InputNeuron(neurons.size()));
					}

					for(size_t i = 0; i < outputSize; i++) {
						addNeuron();
					}

					for(size_t i = 0; i < hiddenUnits; i++) {
						addNeuron();
					}
				};

				Network(const Network &r) : NeuralNetwork::Network(r), neurons(0), _outputsOfNeurons(r._outputsOfNeurons) {
					neurons.push_back(new NeuralNetwork::BiasNeuron());
					for(std::size_t i = 1; i < r.neurons.size(); i++) {
						neurons.push_back(r.neurons[i]->clone());
					}
				}

				Network &operator=(const Network &r);

				/**
				 * @brief Virtual destructor for Network
				 */
				virtual ~Network() {
					for(auto &a:neurons) {
						delete a;
					}
				};

				void reset() {
					for(auto &output: _outputsOfNeurons) {
						output=0.0;
					}
				}

				/**
				 * @brief  This is a function to compute one iterations of network
				 * @param input is input of network
				 * @returns output of network
				 */
				inline virtual std::vector<float> computeOutput(const std::vector<float> &input) override {
					return computeOutput(input, 1);
				}

				/**
				 * @brief  This is a function to compute iterations of network
				 * @param input is input of network
				 * @param iterations is number of iterations
				 * @returns output of network
				 */
				std::vector<float> computeOutput(const std::vector<float> &input, unsigned int iterations);

				std::vector<NeuronInterface *> &getNeurons() {
					return neurons;
				}

				virtual SimpleJSON::Type::Object serialize() const override;

				NeuronInterface &addNeuron() {
					neurons.push_back(new Neuron(neurons.size()));
					NeuronInterface *newNeuron = neurons.back();
					for(std::size_t i = 0; i < neurons.size(); i++) {
						neurons[i]->setInputSize(newNeuron->id + 1);
					}
					return *newNeuron;
				}

				/**
				 * @brief  creates new network from joining two
				 * @param r is network that is connected to outputs of this network
				 * @returns network of constructed from two networks
				 */
				NeuralNetwork::Recurrent::Network connectWith(const NeuralNetwork::Recurrent::Network &r) const;

				static std::unique_ptr<Network> deserialize(const SimpleJSON::Type::Object &);

				std::size_t size() const {
					return neurons.size();
				};

				NeuronInterface &operator[](std::size_t index) {
					return *neurons[index];
				}

				typedef SimpleJSON::Factory<Network> Factory;

			protected:
				std::vector<NeuronInterface *> neurons;
				std::vector<float> _outputsOfNeurons;

			SIMPLEJSON_REGISTER(NeuralNetwork::Recurrent::Network::Factory, NeuralNetwork::Recurrent::Network, deserialize)
		};
	}
}