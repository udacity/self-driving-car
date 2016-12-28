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

#include <math.h>
#include <vector>

#include <string>

#define NEURAL_NETWORK_REGISTER_BASIS_FUNCTION(name,function) SIMPLEJSON_REGISTER(NeuralNetwork::BasisFunction::Factory,name,function)
#define NEURAL_NETWORK_REGISTER_BASIS_FUNCTION_FINISH(name,function) SIMPLEJSON_REGISTER_FINISH(NeuralNetwork::BasisFunction::Factory,name,function)

namespace NeuralNetwork {
namespace BasisFunction {
	class BasisFunction : public SimpleJSON::SerializableObject {
		public:
			virtual ~BasisFunction() {}
			virtual float operator()(const std::vector<float>& weights, const std::vector<float>& input) const =0;

			/**
			 * @brief Function returns clone of object
			 */
			virtual std::unique_ptr<BasisFunction> clone() const = 0;
	};

	typedef SimpleJSON::Factory<BasisFunction> Factory;
}
}