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

#include "Neuron.h"

#include <SimpleJSON/SerializableObject.h>

#include <cstddef>
#include <vector>

#define NEURAL_NETWORK_INIT() const static bool ______TMP= NeuralNetwork::Network::loaded()

namespace NeuralNetwork {

	/**
	* @author Tomas Cernik (Tom.Cernik@gmail.com)
	* @brief Abstract model of simple Network
	*/
	class Network : public SimpleJSON::SerializableObject {
		public:
			/**
			 * @brief Constructor for Network
			 */
			inline Network(std::size_t inputs, std::size_t outputs) : _inputs(inputs), _outputs(outputs) {
				loaded();
			};

			Network(const Network &r) = default;

			/**
			 * @brief Virtual destructor for Network
			 */
			virtual ~Network() { };

			/**
			 * @brief  This is a virtual function for all networks
			 * @param input is input of network
			 * @returns output of network
			 */
			virtual std::vector<float> computeOutput(const std::vector<float> &input) = 0;

			std::size_t inputs() {
				return _inputs;
			}


			std::size_t outputs() {
				return _outputs;
			}

			/**
			 * @param threads is number of threads, if set to 0 or 1 then threading is disabled
			 * @brief Enables or disables Threaded computing of ANN
			 */

			inline virtual void setThreads(const unsigned &threads) final {
				_threads = threads;
			}

		protected:
			/**
			 * @brief Number of threads used by network
			 */
			unsigned _threads = 1;

			std::size_t _inputs;
			std::size_t _outputs;
		public:
			static bool loaded();
	};
}