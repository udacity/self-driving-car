#include <gtest/gtest.h>

#include <SimpleJSON.h>

std::string serialized;

std::vector<float> vec;

TEST(Array,Prepare) {
	for(std::size_t i=0;i<10000;i++) {
		vec.push_back(0.1*i);
	}
}

TEST(Array,Serialize) {
	SimpleJSON::Type::Object obj;
	obj["x"] = vec;
	serialized = obj.serialize();
}

TEST(Array,DeSerialize) {
	SimpleJSON::Type::Object obj = SimpleJSON::parse(serialized).as<SimpleJSON::Type::Object>();
}