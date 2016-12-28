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
#include "Layer.h"

#include <vector>
#include <limits>

namespace NeuralNetwork {
namespace FeedForward {

	/**
	* @author Tomas Cernik (Tom.Cernik@gmail.com)
	* @brief FeedForward model of Artifical neural network
	*/
	class Network: public NeuralNetwork::Network {
		public:

			/**
			 * @brief Constructor for Network
			 * @param _inputSize is number of inputs to network
			 */
			inline Network(size_t _inputSize):NeuralNetwork::Network(_inputSize,_inputSize),layers(),_partialInput(_inputSize+1),_partialOutput(_inputSize+1) {
				appendLayer(_inputSize);
			};

			/**
			 * @brief Virtual destructor for Network
			 */
			virtual ~Network() {
				for(auto &layer:layers) {
					delete layer;
				}
			}

			Layer& appendLayer(std::size_t size=1, const ActivationFunction::ActivationFunction &activationFunction=ActivationFunction::Sigmoid(-4.9)) {
				layers.push_back(new Layer(size,activationFunction));

				if(layers.size() > 1) {
					layers.back()->setInputSize(layers[layers.size() - 2]->size());
				} else {
					_inputs=size;
				}

				if(_partialInput.size() < size+1) {
					_partialInput.resize(size+1);
				}

				if(_partialOutput.size() < size+1) {
					_partialOutput.resize(size+1);
				}

				_outputs=size;

				return *layers.back();
			}

			Layer& operator[](const std::size_t &id) {
				return *layers[id];
			}

			void randomizeWeights();

			std::size_t size() const { return layers.size(); };
			/**
			 * @brief  This is a function to compute one iterations of network
			 * @param input is input of network
			 * @returns output of network
			 */
			virtual std::vector<float> computeOutput(const std::vector<float>& input) override;

			using NeuralNetwork::Network::stringify;

			virtual SimpleJSON::Type::Object serialize() const override {
				std::vector<SimpleJSON::Value> layersSerialized;
				for(std::size_t i=0;i<layers.size();i++) {
					layersSerialized.push_back(layers[i]->serialize());
				}
				return {
					{"class", "NeuralNetwork::FeedForward::Network"},
					{"layers", layersSerialized },
				};
			}

			static std::unique_ptr<Network> deserialize(const SimpleJSON::Type::Object&);

			typedef SimpleJSON::Factory<Network> Factory;

		protected:
			std::vector<Layer*> layers;
			std::vector<float> _partialInput = {};
			std::vector<float> _partialOutput = {};

		private:
			inline Network():NeuralNetwork::Network(0,0),layers() {
			};

		SIMPLEJSON_REGISTER(NeuralNetwork::FeedForward::Network::Factory, NeuralNetwork::FeedForward::Network,NeuralNetwork::FeedForward::Network::deserialize)
	};
}
}