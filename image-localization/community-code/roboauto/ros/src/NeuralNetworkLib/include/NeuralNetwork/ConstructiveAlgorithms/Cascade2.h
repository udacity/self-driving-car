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

#include "../Cascade/Network.h"
#include "../FeedForward/Network.h"
#include "../Learning/QuickPropagation.h"

#include "CascadeCorrelation.h"

#include <random>
#include <algorithm>

// http://fann.cvs.sourceforge.net/viewvc/fann/fann/src/fann_cascade.c?view=markup
// https://github.com/gtomar/cascade

namespace NeuralNetwork {
	namespace ConstructiveAlgorihtms {
		class Cascade2 : public CascadeCorrelation {
			public:
				typedef std::pair<std::vector<float>, std::vector<float>> TrainingPattern;

				Cascade2(std::size_t numberOfCandidate = 18, float maxError = 0.7) : CascadeCorrelation(numberOfCandidate, maxError) {
				}

			protected:

				virtual std::pair<std::shared_ptr<Neuron>, std::vector<float>> trainCandidates(Cascade::Network &network, std::vector<std::shared_ptr<Neuron>> &candidates,
																					   const std::vector<TrainingPattern> &patterns) override;
		};

	}
}