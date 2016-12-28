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
#include "Number.h"
#include "Integer.h"

#include "../Value.h"

#include <string>

namespace SimpleJSON {
	namespace Type {

		class String : public Type {
			public:
				inline String(const std::string &string_=""): string(string_) {
				}

				inline String(std::string &&string_=""): string() {
					std::swap(string,string_);
				}

				inline String(const Number &number): string(std::to_string(number)) {
				}

				inline String(const Integer &number): string(std::to_string(number)) {
				}

				inline virtual std::string serialize(const std::string& prefix="") const override {
					return prefix+"\""+string+"\"";
				}

				inline virtual String* clone() const override {
					return new String(string);
				}

				inline operator const std::string& () const {
					return string;
				}

				inline operator std::string& () {
					return string;
				}

				bool operator==(const String &r) {
					return string == r.string;
				}
			private:
				std::string string;
		};

		inline bool operator==(const String &l, const std::string &r) {
			return l.operator const std::string&()==r;
		}

		inline bool operator==(const String &l, const char *r) {
			return l==std::string(r);
		}
	}

	template<>
	inline std::unique_ptr<Type::Type> Value::toValue<std::string>(const std::string &r) {
		return std::unique_ptr<Type::String>(new Type::String(r));
	}


/*	template<>
	std::unique_ptr<Type::Type> Value::toValue<char const*>(char const* &r) {
		return std::unique_ptr<Type::String>(new Type::String(r));
	}
*/
	template<>
	inline std::string& Value::as<std::string>() {
		return *dynamic_cast<Type::String*>(value);
	}
}