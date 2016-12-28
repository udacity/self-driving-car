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

#include <cmath>

#include "./ActivationFunction.h"

namespace NeuralNetwork {
namespace ActivationFunction {

	/**
	* @author Tomas Cernik (Tom.Cernik@gmail.com)
	* @brief Class for computing sigmoid
	*/
	class Sigmoid: public ActivationFunction {
		public:
			Sigmoid(const float lambdaP = -0.5): lambda(lambdaP) {}


			inline virtual float derivatedOutput(const float &, const float &output) const override { return -lambda*output*(1.0f-output); }
			inline virtual float operator()(const float &x) const override { return 1.0f / (1.0f +exp(lambda*x) ); };

			virtual ActivationFunction* clone() const override {
				return new Sigmoid(lambda);
			}

			virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class", "NeuralNetwork::ActivationFunction::Sigmoid"}, {"lambda", lambda}};
			}

			static std::unique_ptr<Sigmoid> deserialize(const SimpleJSON::Type::Object &obj) {
				return std::unique_ptr<Sigmoid>(new Sigmoid(obj["lambda"].as<double>()));
			}

		protected:
			float lambda;

		NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION(NeuralNetwork::ActivationFunction::Sigmoid, Sigmoid::deserialize)
	};
}
}