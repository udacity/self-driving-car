/*
 *  Copyright 2016 Tomas Cernik, Tom.Cernik@gmail.com
 *  All rights reserved.
 *
 *  This file is part of SimpleJSON
 *
 *  SimpleJSON is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SimpleJSON is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SimpleJSON.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../Value.h"

#include <vector>

namespace SimpleJSON {
	namespace Type {

		class Array : public Type {
			public:
				inline Array() : array() {

				}

				inline Array(const std::initializer_list<SimpleJSON::Value> &list) : array(list) {

				}

				template<typename T>
				inline Array(const std::vector<T> &arr) : array(arr.begin(),arr.end()) {

				}

				inline virtual std::string serialize(const std::string& prefix="") const override {
					std::string serialized="";
					for(std::size_t i=0;i<array.size();i++) {
						if(serialized.length() > 0) {
							serialized+=", ";
						}
						serialized+=array[i].serialize();
					}
					return prefix+"["+serialized+"]";
				}

				inline virtual Array* clone() const override {
					return new Array(array);
				}

				inline SimpleJSON::Value& operator[](std::size_t index) noexcept {
					return array[index];
				}

				inline const SimpleJSON::Value& operator[](std::size_t index) const noexcept {
					return array[index];
				}

				std::size_t size() const {
					return array.size();
				}

				auto begin() {
					return array.begin();
				}

				auto end() {
					return array.end();
				}

				auto begin() const {
					return array.begin();
				}

				auto end() const {
					return array.end();
				}
			private:
				std::vector<SimpleJSON::Value> array;
		};
	}

	template<>
	inline std::unique_ptr<Type::Type> Value::toValue<std::vector<Value>>(const std::vector<Value> &r) {
		return std::unique_ptr<Type::Array>(new Type::Array(r));
	}
	//TODO: quickfix, ifnd better solution
	template<>
	inline std::unique_ptr<Type::Type> Value::toValue<std::vector<float>>(const std::vector<float> &r) {
		return std::unique_ptr<Type::Array>(new Type::Array(r));
	}
}