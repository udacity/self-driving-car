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

#include <vector>

namespace NeuralNetwork {
	namespace ProblemSets {
		typedef std::pair<std::vector<float>, std::vector<float>> TrainingPattern;

		std::vector<TrainingPattern> Parity3(float min = 0.0) {
			return {
				{{min,min,min},{min}},
				{{min,min,1},{1}},
				{{min,1,min},{1}},
				{{min,1,1},{min}},
				{{1,min,min},{1}},
				{{1,min,1},{min}},
				{{1,1,min},{min}},
				{{1,1,1},{1}},
			};
		}

		std::vector<TrainingPattern> Parity4(float min = 0.0) {
			return {
				{{min,min,min,min},{min}},
				{{min,min,min,1},{1}},
				{{min,min,1,min},{1}},
				{{min,min,1,1},{min}},
				{{min,1,min,min},{1}},
				{{min,1,min,1},{min}},
				{{min,1,1,min},{min}},
				{{min,1,1,1},{1}},
				{{1,min,min,min},{1}},
				{{1,min,min,1},{min}},
				{{1,min,1,min},{min}},
				{{1,min,1,1},{1}},
				{{1,1,min,min},{min}},
				{{1,1,min,1},{1}},
				{{1,1,1,min},{1}},
				{{1,1,1,1},{min}},
			};
		}

	}
}