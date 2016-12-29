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

#include "../Neuron.h"

#include <SimpleJSON/SerializableObject.h>

#include <cstddef>
#include <vector>

namespace NeuralNetwork {
namespace FeedForward {

	/**
	 * @author Tomas Cernik (Tom.Cernik@gmail.com)
	 * @brief Class for Layer of FeedForward network
	 */
	class Layer : public SimpleJSON::SerializableObject {

		public:

			Layer(std::size_t size, const ActivationFunction::ActivationFunction &activationFunction):neurons() {
				neurons.push_back(new BiasNeuron);
				for(std::size_t i=0;i<size;i++) {
					neurons.push_back(new Neuron(neurons.size(),activationFunction));
				}
			}

			Layer(const Layer&r):neurons() {
				*this=r;
			}

			Layer& operator=(const Layer &r) {
				for(auto &neuron:neurons) {
					delete neuron;
				}

				neurons.clear();

				for(auto &neuron:r.neurons) {
					neurons.push_back(neuron->clone());
				}
				return *this;
			}

			~Layer() {
				for(auto &neuron:neurons) {
					delete neuron;
				}
			};

			/**
			 * @brief Function adds new neuron
			 * @returns Newly added neuron
			 */
			Neuron& addNeuron(const ActivationFunction::ActivationFunction &activationFunction = ActivationFunction::Sigmoid()) {
				auto neuron = new Neuron(neurons.size(),activationFunction);
				neurons.push_back(neuron);
				return *neuron;
			}

			/**
			 * @brief This is a virtual function for selecting neuron
			 * @param neuron is position in layer
			 * @returns Specific neuron
			 */
			NeuronInterface& operator[](const std::size_t& neuron) {
				return *neurons[neuron];
			}

			/**
			 * @brief This is a virtual function for selecting neuron
			 * @param neuron is position in layer
			 * @returns Specific neuron
			 */
			const NeuronInterface& operator[](const std::size_t& neuron) const {
				return *neurons[neuron];
			}

			void solve(const std::vector<float> &input, std::vector<float> &output);

			/**
			 * @returns Size of layer
			 */
			std::size_t size() const {
				return neurons.size();
			}

			void setInputSize(std::size_t size) {
				for(auto& neuron:neurons) {
					neuron->setInputSize(size);
				}
			}

			virtual SimpleJSON::Type::Object serialize() const override;

			static std::unique_ptr<Layer> deserialize(const SimpleJSON::Type::Object&);

			typedef SimpleJSON::Factory<Layer> Factory;
		protected:
			std::vector<NeuronInterface*> neurons;
		private:
			Layer():neurons() {
			}

		SIMPLEJSON_REGISTER(NeuralNetwork::FeedForward::Layer::Factory, NeuralNetwork::FeedForward::Layer,NeuralNetwork::FeedForward::Layer)
	};
}
}