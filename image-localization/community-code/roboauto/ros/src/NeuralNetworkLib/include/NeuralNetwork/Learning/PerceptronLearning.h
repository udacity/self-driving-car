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
#include <cmath>

#include <NeuralNetwork/FeedForward/Perceptron.h>

namespace NeuralNetwork {
	namespace Learning {

		/** @class PerceptronLearning
		 * @brief Basic algorithm for learning Perceptron
		 */
		class PerceptronLearning {

		public:
			inline PerceptronLearning(FeedForward::Perceptron &perceptronNetwork): perceptron(perceptronNetwork), learningCoefficient(0.1) {
			}

			virtual ~PerceptronLearning() {
			}

			PerceptronLearning(const PerceptronLearning&)=delete;
			PerceptronLearning& operator=(const NeuralNetwork::Learning::PerceptronLearning&) = delete;

			void teach(const std::vector<float> &input, const std::vector<float> &output);

			inline virtual void setLearningCoefficient (const float& coefficient) { learningCoefficient=coefficient; }

		protected:

			FeedForward::Perceptron &perceptron;

			float learningCoefficient;
		};
	}
}