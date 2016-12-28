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

#include "CorrectionFunction.h"

#include <iostream>

namespace NeuralNetwork {
namespace Learning {
namespace CorrectionFunction {
	class ArcTangent : public CorrectionFunction {
		public:
			ArcTangent (const float &c=1.0): coefficient(c) {

			}

			/**
			 * @brief operator returns error for values
			 *
			 */
			inline virtual float operator()(const float &expected, const float &computed) const override final {
				//std::cout << (expected-computed) << ":" << atan(expected-computed) << "\n";
				return atan(coefficient*(expected-computed));
			}
		private:
			const float coefficient;
	};
}
}
}