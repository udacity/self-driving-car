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

#include "./ActivationFunction.h"

#include <cmath>

namespace NeuralNetwork {
namespace ActivationFunction {

	class HyperbolicTangent: public ActivationFunction {
		public:
			HyperbolicTangent(const float& lam=1):lambda(lam) {}

			inline virtual float derivatedOutput(const float &,const float &output) const override { return lambda*(1-output*output); }

			inline virtual float operator()(const float &x) const override { return tanh(lambda*x); }
;
			virtual ActivationFunction* clone() const override {
				return new HyperbolicTangent(lambda);
			}

			virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class", "NeuralNetwork::ActivationFunction::HyperbolicTangent"}, {"lambda", lambda}};
			}

			static std::unique_ptr<HyperbolicTangent> deserialize(const SimpleJSON::Type::Object &obj) {
				return std::unique_ptr<HyperbolicTangent>(new HyperbolicTangent(obj["lambda"].as<double>()));
			}

		protected:
			float lambda;

		NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION(NeuralNetwork::ActivationFunction::HyperbolicTangent, HyperbolicTangent::deserialize)
	};
}
}