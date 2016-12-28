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

#include "Type/Type.h"
#include "Type/Null.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

#include <iostream>

namespace SimpleJSON {

	class SerializableObject;

	class Value {
		public:
			~Value()  {
				if(value !=nullptr) {
					delete value;
				}
			}

			Value() : value (new Type::Null()) {

			}

			Value(const Value &r) : value(r.value->clone()) {

			}

			Value(Value &&r) : value(r.value) {
				r.value=nullptr;
			}

			Value& operator=(Value &&r) {
				std::swap(value, r.value);
				return *this;
			}

			Value& operator=(const Value &r) {
				delete value;
				value=r.value->clone();
				return *this;
			}

			Value(const Value &&r): value(r.value) {

			}

			template<typename T, typename std::enable_if<std::is_base_of<Type::Type, T>::value>::type* = nullptr>
			Value (const T &r) : value(r.clone()) {

			}

			template<typename T>
			static std::unique_ptr<Type::Type> toValue(const T&);

			template<typename T, typename std::enable_if<std::is_same<char[std::extent<T>::value],T>::value>::type* = nullptr>
			Value(const T& r) : value (toValue<std::string>(r).release()) {
			}

			template <typename T,
				typename std::enable_if<
					!std::is_same<char[std::extent<T>::value?std::extent<T>::value:1],T>::value &&
					!std::is_base_of<Type::Type, T>::value &&
					!std::is_base_of<SerializableObject, T>::value
									   >::type* = nullptr>
			Value(const T& r): value(toValue<T>(r).release()) {

			}

			template <typename T,
				typename std::enable_if<
					!std::is_same<char[std::extent<T>::value?std::extent<T>::value:1],T>::value &&
					!std::is_base_of<Type::Type, T>::value &&
					std::is_base_of<SerializableObject, T>::value
									   >::type* = nullptr>
			Value(const T& r): value(r.serialize().clone()) {

			}

			template <typename T,typename std::enable_if<std::is_base_of<Type::Type, T>::value>::type* = nullptr>
			T& as() {
				return *dynamic_cast<T*>(value);
			}

			template <typename T,typename std::enable_if<!std::is_base_of<Type::Type, T>::value>::type* = nullptr>
			T& as();

			template <typename T>
			const T& as() const {
				return const_cast<SimpleJSON::Value*>(this)->as<T>();
			}

			template <typename T, typename std::enable_if<!std::is_base_of<Type::Type, T>::value>::type* = nullptr>
			bool operator==(const T&r) const {
				return as<T>() == r;
			}

			template <typename T>
			operator const T&() const {
				return as<T>();
			}

			template <typename T>
			operator T&() {
				return as<T>();
			}

			template <typename T, typename std::enable_if<std::is_base_of<Type::Type, T>::value>::type* = nullptr>
			bool operator==(const T&r) const {
				return *dynamic_cast<T*>(value) == r;
			}

			template <typename T, typename std::enable_if<std::is_base_of<Type::Type, T>::value>::type* = nullptr>
			bool is() const {
				return dynamic_cast<T*>(value) != nullptr? true:false;
			}

			std::string serialize(const std::string& prefix="") const {
				return value->serialize(prefix);
			}
		protected:
			Type::Type *value;
	};

	inline std::ostream& operator << (std::ostream& str, const SimpleJSON::Value& obj) {
		str << obj.serialize();
		return str;
	}
}