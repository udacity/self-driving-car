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

#include <SimpleJSON.h>

#include <sstream>
#include <string>

namespace SimpleJSON {
	class SerializableObject {
		public:

			inline virtual ~SerializableObject() {
			}

			virtual SimpleJSON::Type::Object serialize() const = 0;

			inline virtual std::string stringify() const final;
	};

	inline static std::ostream& operator<<(std::ostream& stream, const SerializableObject &object) {
		const SimpleJSON::Type::Object obj=object.serialize();
		stream << obj.serialize();

		return stream;
	}
}

std::string SimpleJSON::SerializableObject::stringify() const{
	std::ostringstream s;
	operator<<(s,*this);
	return s.str();
}