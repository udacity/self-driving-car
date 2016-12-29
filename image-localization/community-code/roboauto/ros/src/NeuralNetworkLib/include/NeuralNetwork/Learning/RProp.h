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

		/** @class Resilient Propagation
		 * @brief
		 */
		class RProp : public BatchPropagation {

			public:
				RProp(FeedForward::Network &feedForwardNetwork, std::shared_ptr<CorrectionFunction::CorrectionFunction> correction = std::make_shared<CorrectionFunction::Linear>()):
					BatchPropagation(feedForwardNetwork, correction) {
				}

				RProp(const RProp&)=delete;
				RProp& operator=(const NeuralNetwork::Learning::RProp&) = delete;

				void setInitialWeightChange(float initVal) {
					initialWeightChange=initVal;
				}
				void setLearningCoefficient(float) {

				}
			protected:

				virtual inline void resize() override {
					BatchPropagation::resize();

					_lastGradients =_gradients;

					_changesOfWeightChanges = _lastGradients;
					for(std::size_t i = 1; i < _network.size(); i++) {
						for(std::size_t j = 0; j < _changesOfWeightChanges[i].size(); j++) {
							std::fill(_changesOfWeightChanges[i][j].begin(),_changesOfWeightChanges[i][j].end(),initialWeightChange);
						}
					}
					_lastWeightChanges = _lastGradients;
					for(std::size_t i = 1; i < _network.size(); i++) {
						for(std::size_t j = 0; j < _lastWeightChanges[i].size(); j++) {
							std::fill(_lastWeightChanges[i][j].begin(),_lastWeightChanges[i][j].end(),0.1);
						}
					}
				}

				void updateWeightsAndEndBatch() override;

				std::vector<std::vector<std::vector<float>>> _lastGradients = {};
				std::vector<std::vector<std::vector<float>>> _lastWeightChanges = {};
				std::vector<std::vector<std::vector<float>>> _changesOfWeightChanges = {};

				float maxChangeOfWeights = 50;
				float minChangeOfWeights = 0.0001;

				float initialWeightChange=0.02;
				float weightChangePlus=1.2;
				float weightChangeMinus=0.5;
		};
	}
}