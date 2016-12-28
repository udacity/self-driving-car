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

#include <SimpleJSON/Type/Boolean.h>
#include <SimpleJSON/Type/Integer.h>
#include <SimpleJSON/Type/Number.h>
#include <SimpleJSON/Type/Object.h>
#include <SimpleJSON/Type/String.h>
#include <SimpleJSON/Type/Array.h>
#include <SimpleJSON/Value.h>

#include <string>
#include <memory>

namespace SimpleJSON {
	class JSONParser {
		public:
			static Value parse(const std::string& str) {
				FastString s(str);
				return std::move(parseValue(s));
			}

			inline static SimpleJSON::Type::Object parseObject(const std::string& str) {
				FastString s(str);
				return std::move(parseObject(s));
			}

		private:
			class FastString {
				public:
					FastString(const std::string&s) : str(s) {

					}

					FastString (const FastString&) = delete;
					void operator=(const FastString&) = delete;

					auto operator[](std::size_t ind) const {
						return str[ind + baseIndex];
					}

					auto length() const {
						return str.length()-baseIndex;
					}

					void shift(std::size_t ind) {
						baseIndex+=ind;
					}

					auto substr(std::size_t from, std::size_t length) const {
						return str.substr(baseIndex+from,length);
					}

					auto c_str() const {
						return str.c_str()+baseIndex;
					}
				private:
					const std::string &str;
					std::size_t baseIndex = 0;
			};

			static bool isFloat(const FastString &s) {
				std::size_t i =0;
				while(1) {
					if(std::isdigit(s[i]) || s[i]=='-') {
						i++;
					}else if(s[i]=='.') { // TODO
						return true;
					}else {
						return false;
					}
				}
			}
			static void removeWhiteSpaces(FastString& str) {
				std::size_t ind =0;
				while(std::isspace(str[ind])) {
					ind++;
				}
				if(ind > 0) {
					str.shift(ind);
				}
			}
			inline static SimpleJSON::Type::Object parseObject(FastString& str);

			inline static SimpleJSON::Type::Array parseArray(FastString& str);

			inline static SimpleJSON::Type::String parseString(FastString& str);

			inline static SimpleJSON::Type::Number parseNumber(FastString& str);

			inline static SimpleJSON::Type::Null parseNull(FastString& str);

			inline static SimpleJSON::Type::Boolean parseBool(FastString& str);

			inline static Value parseValue(FastString& str);

			inline static SimpleJSON::Type::Integer parseInteger(FastString &str);
	};

	inline Value parse(const std::string &str) {
		return std::move(JSONParser::parse(str));
	}

	SimpleJSON::Type::Object JSONParser::parseObject(FastString& str) {
		Type::Object obj;
		bool inObject=true;
		removeWhiteSpaces(str);
		str.shift(1);
		do {
			removeWhiteSpaces(str);
			if(str[0] == '}') {
				inObject=false;
				str.shift(1);
			}else {
				if(str[0]==',') {
					str.shift(1);
				}
				std::string key = parseString(str);
				removeWhiteSpaces(str);
				str.shift(1);
				obj[key] = parseValue(str);
			}
		}
		while(inObject);
		return std::move(obj);
	}

	SimpleJSON::Type::Array JSONParser::parseArray(FastString& str) {
		std::vector<Value> array;
		bool inArray=true;
		do {
			removeWhiteSpaces(str);
			if(str[0]==']') {
				str.shift(1);
				inArray=false;
			}else {
				str.shift(1);
				array.push_back(parseValue(str));
			}
		}while(inArray);
		return std::move(array);
	}

	SimpleJSON::Type::String JSONParser::parseString(FastString& str) {
		removeWhiteSpaces(str);

		std::size_t end=1;
		while(str[end]!= '"') {
			if(str[end]=='\\') { // TODO: http://www.json.org/
				end+=2;
			}else {
				end++;
			}
		}

		std::string tmp;
		for(std::size_t i=1;i<end;i++) {
			if(str[i]=='\\') { // TODO
				tmp+=str[i+1];
				i++;
			} else {
				tmp+=str[i];
			}
		}

		str.shift(end+1);
		return tmp;
	}

	SimpleJSON::Type::Number JSONParser::parseNumber(FastString& str) {
		removeWhiteSpaces(str);
		std::size_t nextChar=0;
		//double ret=std::stod(str,&nextChar);
		double ret = __gnu_cxx::__stoa(&std::strtod, "stod", str.c_str(), &nextChar);

		str.shift(nextChar);
		return ret;
	}

	SimpleJSON::Type::Integer JSONParser::parseInteger(FastString &str) {
		removeWhiteSpaces(str);
		std::size_t nextChar=0;
		//int ret=std::stoi(str,&nextChar);
		int ret = __gnu_cxx::__stoa<long, int>(&std::strtol, "stoi", str.c_str(), &nextChar, 10);
		str.shift(nextChar);
		return ret;
	}

	SimpleJSON::Type::Null JSONParser::parseNull(FastString& str) {
		removeWhiteSpaces(str);
		str.shift(4);
		return {};
	}

	SimpleJSON::Type::Boolean JSONParser::parseBool(FastString& str) {
		removeWhiteSpaces(str);
		if(str.length() >= 4 && str.substr(0,4) == "true") {
			str.shift(4);
			return true;
		} else {
			str.shift(5);
			return false;
		}
	}

	Value JSONParser::parseValue(FastString &str) {
		removeWhiteSpaces(str);
		if(str[0] == '{') {
			return parseObject(str);
		}else if(str[0] == '[') {
			return parseArray(str);
		}else if(str[0] == '\"') {
			return parseString(str);
		}else if(std::isdigit(str[0]) || str[0]=='-' || str[0]=='.') { // TODO init
			if(isFloat(str)) {
				return parseNumber(str);
			}else {
				return parseInteger(str);
			}

		}else if( str[0]=='n') {
			return parseNull(str);
		}else {
			return parseBool(str);
		}
	}

}
