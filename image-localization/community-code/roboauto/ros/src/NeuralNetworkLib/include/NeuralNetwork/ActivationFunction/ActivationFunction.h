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

#include <SimpleJSON/SerializableObject.h>
#include <SimpleJSON/Factory.h>

#define NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION(name,function) SIMPLEJSON_REGISTER(NeuralNetwork::ActivationFunction::Factory,name,function)

#define NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION_FINISH(name,function) SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::ActivationFunction::Factory,name,function)

namespace NeuralNetwork {
namespace ActivationFunction {

	/**
	* @author Tomas Cernik (Tom.Cernik@gmail.com)
	* @brief Abstract class of activation function
	*/
	class ActivationFunction : public SimpleJSON::SerializableObject {
		public:

			virtual ~ActivationFunction() {}

			/**
			 * @brief Returns derivation of output, it is slower than version with output as it needs to compute output
			 * @param input is input of function
			 */
			inline float derivatedOutput(const float &input) const {
				return derivatedOutput(input,operator()(input));
			};

			/**
			 * @brief Returns derivation of output
			 * @param input is input of function
			 * @param output is output of function
			 * @see derivatedOutput
			 */
			virtual float derivatedOutput(const float &input, const float &output) const=0;

			/**
			 * @brief Returns value of output
			 * @param x is input of function
			 */
			virtual float operator()(const float &x) const=0;

			/**
			 * @brief Function returns clone of object
			 */
			virtual ActivationFunction* clone() const = 0;

	};

	typedef SimpleJSON::Factory<ActivationFunction> Factory;

}
}