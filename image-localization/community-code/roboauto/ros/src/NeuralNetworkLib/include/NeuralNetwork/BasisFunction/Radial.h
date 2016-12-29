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

namespace NeuralNetwork
{
namespace BasisFunction
{
	class Radial: public BasisFunction
	{
		public:
			Radial() {}

			virtual float operator()(const std::vector<float>& weights, const std::vector<float>& input) const override {
				float sum = 0.0;
				for(std::size_t i=0;i<weights.size();i++) {
					sum+=pow(input[i]-weights[i],2);
				}
				return sqrt(sum);
			}

			virtual std::unique_ptr<BasisFunction> clone() const override {
				return std::unique_ptr<BasisFunction>(new Radial());
			}

			virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class", "NeuralNetwork::BasisFunction::Radial"}};
			}

			static std::unique_ptr<Radial> deserialize(const SimpleJSON::Type::Object &) {
				return std::unique_ptr<Radial>(new Radial());
			}

		NEURAL_NETWORK_REGISTER_BASIS_FUNCTION(NeuralNetwork::BasisFunction::Radial, deserialize)
	};
}
}