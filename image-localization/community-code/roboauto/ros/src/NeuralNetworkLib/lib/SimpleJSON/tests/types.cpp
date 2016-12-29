#include <SimpleJSON.h>

#include <gtest/gtest.h>

TEST(Integers,CMP) {
	ASSERT_EQ(SimpleJSON::Type::Integer(5),5);
	ASSERT_NE(SimpleJSON::Type::Integer(5),6);
}

TEST(Number,CMP) {
	ASSERT_EQ(SimpleJSON::Type::Number(5), 5);
	ASSERT_EQ(SimpleJSON::Type::Number(5.0), 5.0);
	ASSERT_NE(SimpleJSON::Type::Number(5.0), 6.0);
}

TEST(Boolean,CMP) {
	ASSERT_EQ(SimpleJSON::Type::Boolean(true), true);
	ASSERT_EQ(SimpleJSON::Type::Boolean(false), false);
	ASSERT_NE(SimpleJSON::Type::Boolean(true), false);
}

int testInt(const int& int_) {
	return int_;
}

int testNumber(const double& d_) {
	return d_;
}

TEST(NumericConversions,CMP) {

	ASSERT_EQ(SimpleJSON::Type::Number(5.0), SimpleJSON::Type::Integer(5));
	ASSERT_EQ(testNumber(SimpleJSON::Type::Number(5.0)), SimpleJSON::Type::Integer(5));
	ASSERT_EQ(SimpleJSON::Type::Number(5.0), testNumber(SimpleJSON::Type::Integer(5)));
}

TEST(String,CMP) {
	ASSERT_EQ(SimpleJSON::Type::String("hello"), "hello");
}

TEST(StringConvertible,Integer) {
	ASSERT_EQ(SimpleJSON::Type::Integer(5), SimpleJSON::Type::String("5"));
}

TEST(Array,Numeric) {
	auto array= SimpleJSON::Type::Array({1,2,3,4,5});
	ASSERT_EQ(array[0], 1);
	ASSERT_EQ(array[1], 2);
	std::vector<float> tmp = {1,2,3,4,5};
	auto array2= SimpleJSON::Type::Array(tmp);
	ASSERT_EQ(array2[0], 1.0);
	ASSERT_EQ(array2[1], 2.0);
}

TEST(Array,String) {
	std::vector<std::string> tmp = {"1","2","3","4","5"};
	auto array2= SimpleJSON::Type::Array(tmp);
	ASSERT_EQ(array2[0], std::string("1"));
	ASSERT_EQ(array2[1], std::string("2"));
}

TEST(Object,Numeric) {
	auto object= SimpleJSON::Type::Object({{"a",1},{"b",2},{"c",3}});
	ASSERT_EQ(object["a"], 1);
}

TEST(Object,String) {
	auto object= SimpleJSON::Type::Object({{"a","A"},{"b","B"},{"c","C"}});
	ASSERT_EQ(object["a"], std::string("A"));
}