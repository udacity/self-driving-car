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

#include "BatchPropagation.h"

namespace NeuralNetwork {
namespace Learning {

	/** @class BackPropagation
	 * @brief 
	 */
	class BackPropagation : public BatchPropagation {

		public:
			BackPropagation(FeedForward::Network &feedForwardNetwork, std::shared_ptr<CorrectionFunction::CorrectionFunction> correction = std::make_shared<CorrectionFunction::Linear>()):
				BatchPropagation(feedForwardNetwork,correction), learningCoefficient(0.4) {
				resize();
			}

			BackPropagation(const BackPropagation&)=delete;
			BackPropagation& operator=(const NeuralNetwork::Learning::BackPropagation&) = delete;

			void setLearningCoefficient (const float& coefficient) {
				learningCoefficient=coefficient;
			}

			float getMomentumWeight() const {
				return momentumWeight;
			}

			void setMomentumWeight(const float& m) {
				momentumWeight=m;
				resize();
			}

			float getWeightDecay() const {
				return weightDecay;
			}

			void setWeightDecay(const float& wd) {
				weightDecay=wd;
			}

		protected:

			virtual inline void resize() override {
				BatchPropagation::resize();
				if(momentumWeight > 0.0) {
					_lastDeltas = _gradients;
				}
			}

			virtual void updateWeightsAndEndBatch() override;

			float learningCoefficient;
			float momentumWeight = 0.0;
			float weightDecay = 0.0;

			std::vector<std::vector<std::vector<float>>> _lastDeltas = {};

	};
}
}