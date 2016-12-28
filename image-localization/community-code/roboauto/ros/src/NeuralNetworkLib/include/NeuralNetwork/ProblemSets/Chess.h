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
#include <random>

namespace NeuralNetwork {
	namespace ProblemSets {
		typedef std::pair<std::vector<float>, std::vector<float>> TrainingPattern;

		std::vector<TrainingPattern> Chess3X3(float min, std::size_t patterns) {
			std::vector<TrainingPattern> ret;
			std::mt19937 _generator(rand());
			std::uniform_real_distribution<> _distribution(min,1);
			float step = (1.0-min)/3.0;
			for(std::size_t i=0;i<patterns;i++) {
				float x=_distribution(_generator);
				float y=_distribution(_generator);
				int classX= (static_cast<int>((x-min)/step) + static_cast<int>((y-min)/step)) % 2;
				if(classX == 0) {
					ret.push_back({{x,y},{min}});
				} else {
					ret.push_back({{x,y},{1.0}});
				}
			}
			return ret;
		}

	}
}