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

namespace SimpleJSON {
	namespace Type {

		class Object : public Type {
			public:
				inline Object() : map() {

				}

				inline Object(std::initializer_list<std::pair<std::string const, Value>> list) : map() {
					for(const auto&it :list) {
						map.insert(it);
					}
				}

				inline virtual std::string serialize(const std::string& prefix="") const override {
					std::string serialized="";
					for(auto it :map) {
						serialized+=(serialized.length() > 0? ", \"" : "\"")+it.first+ "\" : " +it.second.serialize();
					}
					return prefix+"{"+serialized+"}";
				}

				inline virtual Object* clone() const override {
					return new Object(*this);
				}

				inline auto find(const std::string &key) const {
					return map.find(key);
				}

				template <typename T>
				inline SimpleJSON::Value& operator[](T a) {
					return map[a];
				}

				template <typename T>
				inline const SimpleJSON::Value& operator[](T a) const {
					return map.find(a)->second;
				}

				inline auto end() const {
					return map.end();
				}
			private:
				std::map<std::string,SimpleJSON::Value> map;
		};

	}
}