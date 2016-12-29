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

#include "./BasisFunction.h"

namespace NeuralNetwork {
namespace BasisFunction {

	class Product: public BasisFunction {
		public:
			Product() {}

			/**
			 * @brief function computes product of inputs, where weight > 0.5
			 */
			inline virtual float operator()(const std::vector<float>& weights, const std::vector<float>& input) const override {
				float product=1.0;
				for(size_t i=0;i<weights.size();i++) {
					if(weights[i] > 0.5)
						product=product*input[i];
				}
				return product;
			}

			virtual std::unique_ptr<BasisFunction> clone() const override {
				return std::unique_ptr<BasisFunction>(new Product());
			}

			virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class", "NeuralNetwork::BasisFunction::Product"}};
			}

			static std::unique_ptr<Product> deserialize(const SimpleJSON::Type::Object &) {
				return std::unique_ptr<Product>(new Product());
			}

		NEURAL_NETWORK_REGISTER_BASIS_FUNCTION(NeuralNetwork::BasisFunction::Product, deserialize)
	};
}
}