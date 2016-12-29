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

#include <NeuralNetwork/ActivationFunction/Sigmoid.h>
#include <NeuralNetwork/BasisFunction/Linear.h>

#include <string>
#include <vector>

#include <limits>

namespace NeuralNetwork
{
	/**
	 * @author Tomas Cernik (Tom.Cernik@gmail.com)
	 * @brief Abstract class of neuron. All Neuron classes should derive from this on
	 */
	class NeuronInterface : public SimpleJSON::SerializableObject {
		public:
			NeuronInterface(const unsigned long &_id=0): id(_id), weights(1),_output(1),
														 _value(0) {

			}

			NeuronInterface(const NeuronInterface &r): id(r.id), weights(r.weights),_output(r._output),
													   _value(r._value) {
				weights=weights;
			}

			/**
			 * @brief virtual destructor for Neuron
			 */
			virtual ~NeuronInterface() {};

			const std::vector<float> & getWeights() const {
				return weights;
			}

			void setWeights(const std::vector<float> &weights_) {
				weights=weights_;
			}

			/**
			 * @brief getter for neuron weight
			 * @param &neuron is neuron it's weight is returned
			 */
			inline virtual float weight(const NeuronInterface &neuron) const final {
				return weights[neuron.id];
			}

			/**
			 * @brief getter for neuron weight
			 * @param &neuronID is id of neuron
			 */
			inline virtual float weight(const std::size_t &neuronID) const final {
				return weights[neuronID];
			}

			/**
			 * @brief This is a virtual function for storing network
			 * @returns json describing network and it's state
			 */
			inline virtual float& weight(const NeuronInterface &neuron) final {
				return weights[neuron.id];
			}

			/**
			 * @brief getter for neuron weight
			 * @param neuronID is id of neuron
			 */
			inline virtual float& weight(const std::size_t &neuronID) final {
				return weights[neuronID];
			}

			/**
			 * @brief Returns output of neuron
			 */
			inline virtual float output() const final {
				return _output;
			}
			/**
			 * @brief Returns input of neuron
			 */
			inline virtual float value() const final {
				return _value;
			};

			virtual float operator()(const std::vector<float>& inputs) =0;

			/**
			 * @brief function resizes weighs to desired size
			 */
			inline virtual void setInputSize(const std::size_t &size) final {
				if(weights.size()<size) {
					weights.resize(size);
				}
			}

			/**
			 * @brief Function returns clone of object
			 */
			virtual NeuronInterface* clone() const = 0;

			/**
			 * @brief getter for basis function of neuron
			 */
			virtual BasisFunction::BasisFunction& getBasisFunction() =0;

			/**
			 * @brief getter for activation function of neuron
			 */
			virtual const ActivationFunction::ActivationFunction& getActivationFunction() const =0;

			virtual void setBasisFunction(const BasisFunction::BasisFunction& basisFunction) =0;

			virtual void setActivationFunction(const ActivationFunction::ActivationFunction &activationFunction) =0;

			/**
			 * @brief id is identificator if neuron
			 */
			const unsigned long id;

			typedef SimpleJSON::Factory<NeuronInterface> Factory;
		protected:
			std::vector<float> weights;
			float _output;
			float _value;
	};

	/**
	 * @author Tomas Cernik (Tom.Cernik@gmail.com)
	 * @brief Class of FeedForward neuron.
	 */
	class Neuron: public NeuronInterface {
		public:
			Neuron(unsigned long _id=0, const ActivationFunction::ActivationFunction &activationFunction=ActivationFunction::Sigmoid(-4.9)):
										NeuronInterface(_id), basis(new BasisFunction::Linear),
										activation(activationFunction.clone()) {
				_output=0.0;
			}

			Neuron(const Neuron &r): NeuronInterface(r), basis(r.basis->clone().release()), activation(r.activation->clone()) {
			}

			virtual ~Neuron() {
				delete basis;
				delete activation;
			};

			Neuron operator=(const Neuron&) = delete;

			float operator()(const std::vector<float>& inputs) {
				//compute value
				_value=basis->operator()(weights,inputs);

				//compute output
				_output=activation->operator()(_value);

				return _output;
			}

			virtual Neuron* clone() const override {
				Neuron *n = new Neuron(*this);
				return n;
			}

			virtual BasisFunction::BasisFunction& getBasisFunction() override {
				return *basis;
			}

			virtual const ActivationFunction::ActivationFunction& getActivationFunction() const override {
				return *activation;
			}

			virtual void setBasisFunction(const BasisFunction::BasisFunction& basisFunction) override {
				delete basis;
				basis=basisFunction.clone().release();

			}

			virtual void setActivationFunction(const ActivationFunction::ActivationFunction &activationFunction) override {
				delete activation;
				activation=activationFunction.clone();
			}

			virtual SimpleJSON::Type::Object serialize() const override;

			static std::unique_ptr<Neuron> deserialize(const SimpleJSON::Type::Object &obj);
		protected:

			BasisFunction::BasisFunction *basis;
			ActivationFunction::ActivationFunction *activation;

		SIMPLEJSON_REGISTER(NeuralNetwork::NeuronInterface::Factory, NeuralNetwork::Neuron,NeuralNetwork::Neuron::deserialize)
	};

	class BiasNeuron: public NeuronInterface {
		public:
			class usageException : public std::exception {
				public:
					usageException(const std::string &text_):text(text_) {

					}

					virtual const char* what() const noexcept override {
						return text.c_str();
					}
				protected:
					std::string text;
			};

			virtual float operator()(const std::vector< float >&) override { return 1.0; }

			virtual BiasNeuron* clone() const { return new BiasNeuron(); }

			virtual BasisFunction::BasisFunction& getBasisFunction() override {
				throw usageException("basis function");
			}

			virtual const ActivationFunction::ActivationFunction& getActivationFunction() const override {
				throw usageException("biasNeuron - activation function");
			}

			virtual void setBasisFunction(const BasisFunction::BasisFunction&) override {
				throw usageException("basis function");

			}

			virtual void setActivationFunction(const ActivationFunction::ActivationFunction &) override {
				throw usageException("activation function");
			}

			virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class", "NeuralNetwork::BiasNeuron"}};
			}

			static std::unique_ptr<BiasNeuron> deserialize(const SimpleJSON::Type::Object &) {
				return std::unique_ptr<BiasNeuron>(new BiasNeuron());
			}

		SIMPLEJSON_REGISTER(NeuralNetwork::NeuronInterface::Factory, NeuralNetwork::BiasNeuron,NeuralNetwork::BiasNeuron::deserialize)
	};

	class InputNeuron: public NeuronInterface {
		public:
			class usageException : public std::exception {
				public:
					usageException(const std::string &text_):text(text_) {

					}

					virtual const char* what() const noexcept override {
						return text.c_str();
					}
				protected:
					std::string text;
			};

			InputNeuron(long unsigned int _id): NeuronInterface(_id) {
				
			}

			virtual float operator()(const std::vector< float >&) override { return 1.0; }

			virtual InputNeuron* clone() const { return new InputNeuron(id); }

			virtual BasisFunction::BasisFunction& getBasisFunction() override {
				throw usageException("basis function");
			}

			virtual const ActivationFunction::ActivationFunction& getActivationFunction() const override {
				throw usageException("input neuron - activation function");
			}

			virtual void setBasisFunction(const BasisFunction::BasisFunction&) override {
				throw usageException("basis function");

			}

			virtual void setActivationFunction(const ActivationFunction::ActivationFunction &) override {
				throw usageException("activation function");
			}

			virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class", "NeuralNetwork::InputNeuron"}, {"id", id}};
			}

			static std::unique_ptr<NeuronInterface> deserialize(const SimpleJSON::Type::Object &obj) {
				return std::unique_ptr<NeuronInterface>(new InputNeuron(obj["id"].as<int>()));
			}

		SIMPLEJSON_REGISTER(NeuralNetwork::NeuronInterface::Factory, NeuralNetwork::InputNeuron,NeuralNetwork::InputNeuron::deserialize)
	};
}