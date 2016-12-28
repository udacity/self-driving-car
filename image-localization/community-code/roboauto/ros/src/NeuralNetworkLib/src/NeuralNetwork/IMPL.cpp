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

#include <NeuralNetwork/ActivationFunction/Heaviside.h>
#include <NeuralNetwork/ActivationFunction/Linear.h>
#include <NeuralNetwork/ActivationFunction/HyperbolicTangent.h>
#include <NeuralNetwork/ActivationFunction/Sigmoid.h>

NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION_FINISH(NeuralNetwork::ActivationFunction::Heaviside, Heaviside::deserialize)
NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION_FINISH(NeuralNetwork::ActivationFunction::HyperbolicTangent, HyperbolicTangent::deserialize)
NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION_FINISH(NeuralNetwork::ActivationFunction::Linear, Linear::deserialize)
NEURAL_NETWORK_REGISTER_ACTIVATION_FUNCTION_FINISH(NeuralNetwork::ActivationFunction::Sigmoid, Sigmoid::deserialize)

#include <NeuralNetwork/BasisFunction/Linear.h>
#include <NeuralNetwork/BasisFunction/Product.h>
#include <NeuralNetwork/BasisFunction/Radial.h>

NEURAL_NETWORK_REGISTER_BASIS_FUNCTION_FINISH(NeuralNetwork::BasisFunction::Linear, deserialize)
NEURAL_NETWORK_REGISTER_BASIS_FUNCTION_FINISH(NeuralNetwork::BasisFunction::Product, deserialize)
NEURAL_NETWORK_REGISTER_BASIS_FUNCTION_FINISH(NeuralNetwork::BasisFunction::Radial, deserialize)


#include <NeuralNetwork/Network.h>

bool NeuralNetwork::Network::loaded() {
	return true;
}