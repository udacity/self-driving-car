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

#include <NeuralNetwork/FeedForward/Network.h>
#include "BatchPropagation.h"

namespace NeuralNetwork {
	namespace Learning {

		/** @class QuickPropagation
		 * @brief
		 */
		class QuickPropagation : public BatchPropagation {

		public:
			inline QuickPropagation(FeedForward::Network &feedForwardNetwork, std::shared_ptr<CorrectionFunction::CorrectionFunction> correction = std::make_shared<CorrectionFunction::Linear>()):
				BatchPropagation(feedForwardNetwork,correction) {
			}

			virtual ~QuickPropagation() {
			}

			void setLearningCoefficient (const float& coefficient) {
			}

			protected:

			virtual void updateWeightsAndEndBatch() override;

			float _maxChange=1.75;

			virtual inline void resize() override {
				BatchPropagation::resize();
				_lastGradients = _gradients;
				_lastDeltas = _gradients;
			}

			std::vector<std::vector<std::vector<float>>> _lastDeltas = {};
			std::vector<std::vector<std::vector<float>>> _lastGradients = {};
		};
	}
}