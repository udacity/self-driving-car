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
#include "../Value.h"

#include <string>

namespace SimpleJSON {
	namespace Type {

		class Boolean : public Type {
			public:
				Boolean(const bool &val=false): value(val) {

				}

				virtual std::string serialize(const std::string& prefix="") const override {
					return prefix+std::to_string(value);
				}

				virtual Boolean* clone() const override {
					return new Boolean(value);
				}

				operator bool() const {
					return value;
				}

				operator bool&() {
					return value;
				}
			private:
				bool value;
		};

	}

	template<>
	inline std::unique_ptr<Type::Type> Value::toValue<bool>(const bool &r) {
		return std::unique_ptr<Type::Boolean>(new Type::Boolean(r));
	}

	template <>
	inline bool& Value::as<bool>() {
		return *dynamic_cast<Type::Boolean*>(value);
	}

}