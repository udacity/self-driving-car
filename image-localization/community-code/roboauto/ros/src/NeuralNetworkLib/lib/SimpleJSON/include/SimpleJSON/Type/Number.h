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

#include "Type.h"

#include <string>

namespace SimpleJSON {
	namespace Type {

		class Number : public Type {
			public:
				Number(const double &number=0.0): num(number) {

				}

				virtual std::string serialize(const std::string& prefix="") const override {
					return prefix+std::to_string(num);
				}

				virtual Number* clone() const override {
					return new Number(num);
				}

				operator double () const noexcept {
					return num;
				}

				operator double& () noexcept {
					return num;
				}

			private:
				double num;
		};

	}

	template<>
	inline std::unique_ptr<Type::Type> Value::toValue<double>(const double &r) {
		return std::unique_ptr<Type::Number>(new Type::Number(r));
	}

	template<>
	inline std::unique_ptr<Type::Type> Value::toValue<float>(const float &r) {
		return std::unique_ptr<Type::Number>(new Type::Number(r));
	}

	template <>
	inline double& Value::as<double>() {
		return *dynamic_cast<Type::Number*>(value);
	}
}