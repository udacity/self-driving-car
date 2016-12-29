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

#include <memory>
#include <functional>
#include <iostream>

#define SIMPLEJSON_REGISTER(FACTORY,NAME,FUNCTION) private: static const bool __SIMPLE_JSON_REGISTERED;

#define SIMPLEJSON_REGISTER_FINISH(FACTORY,NAME,FUNCTION) const bool NAME::__SIMPLE_JSON_REGISTERED = FACTORY::registerCreator( #NAME ,FUNCTION);

namespace SimpleJSON {

	template <typename T>
	/**
	 * @brief Factory for creating objects
	 */
	class Factory {
		public:

			class Exception : public std::exception {
				public:
					Exception(const std::string& what) : what_(what) {

					}
					virtual const char* what() const noexcept {
						return what_.c_str();
					}
				protected:
					std::string what_;
			};

			class Redefined : public Exception {
				public:
					Redefined(const std::string& className) : Exception("Constructor for "+ className+" redefined") {

					}
			};

			class ClassNotFound : public Exception {
				public:
					ClassNotFound(const std::string& className) : Exception("Class \""+className+"\" can not be destringified.") {

					}
			};

			typedef std::function<std::unique_ptr<T>(const SimpleJSON::Type::Object&) > CreatorFunction;

			Factory(const Factory&) = delete;

			static bool registerCreator(const std::string &className, CreatorFunction c) {
				return instance().registerCreator_(className, c);
			}

			static std::unique_ptr<T> deserialize(const std::string &str) {
				SimpleJSON::Value v = SimpleJSON::parse(str);
				return instance().deserialize_(v.as<Type::Object>());
			}

			static std::unique_ptr<T> deserialize(const SimpleJSON::Type::Object& object) {
				return instance().deserialize_(object);
			}

			// Thread safe as of C++11
			static Factory<T>& instance() {
				static Factory<T> fact;
				return fact;
			}

		private:

			inline bool registerCreator_(const std::string &className, CreatorFunction f);

			inline std::unique_ptr<T> deserialize_ (const SimpleJSON::Type::Object& object);

			Factory(): creators() {

			}

			std::map<std::string,CreatorFunction> creators;
	};
}

template<typename T>
bool SimpleJSON::Factory<T>::registerCreator_(const std::string &className, CreatorFunction f) {
	const auto& creator=creators.find(className);
	if(creator != creators.end()) {
		throw Redefined(className);
	}

	creators[className] = f;
	return true;
}

template<typename T>
std::unique_ptr<T> SimpleJSON::Factory<T>::deserialize_(const SimpleJSON::Type::Object& object) {
	std::string className = object["class"].as<std::string>();

	const auto& creator=creators.find(className);
	if(creator == creators.end()) {
		throw ClassNotFound(className);
	}

	return creator->second(object);
}