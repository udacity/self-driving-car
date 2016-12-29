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

#include <NeuralNetwork/FeedForward/Network.h>

#include "CorrectionFunction/Linear.h"

#include <vector>
#include <memory>

namespace NeuralNetwork {
namespace Learning {
	class BatchPropagation {
		public:
			BatchPropagation(FeedForward::Network &ffn, std::shared_ptr<CorrectionFunction::CorrectionFunction> correction) : _network(ffn), _correctionFunction(correction) {

			}

			virtual ~BatchPropagation() {

			}

			void teach(const std::vector<float> &input, const std::vector<float> &output);

			void finishTeaching();

			std::size_t getBatchSize() const {
				return _batchSize;
			}

			void setBatchSize(std::size_t size) {
				_batchSize = size;
			}
		protected:
			virtual void updateWeightsAndEndBatch() = 0;
			virtual void resize();

			FeedForward::Network &_network;
			std::shared_ptr<CorrectionFunction::CorrectionFunction> _correctionFunction;

			std::size_t _batchSize = 1;
			std::size_t _currentBatchSize = 0;

			std::vector<std::vector<float>> _slopes = {};
			std::vector<std::vector<std::vector<float>>> _gradients = {};

			bool init = false;
		private:
			void computeSlopes(const std::vector<float> &expectation);
			void computeDeltas(const std::vector<float> &input);
	};
}
}