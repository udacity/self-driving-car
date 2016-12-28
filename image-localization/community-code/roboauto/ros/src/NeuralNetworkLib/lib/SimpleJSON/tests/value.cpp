#include <SimpleJSON.h>

#include <gtest/gtest.h>

TEST(VAlue,Integer) {
	ASSERT_EQ(SimpleJSON::Value(5),SimpleJSON::Type::Integer(5));
	ASSERT_EQ(SimpleJSON::Value(5),5);
	SimpleJSON::Value val(4);
	val=5;
	ASSERT_EQ(val,5);
	ASSERT_EQ(SimpleJSON::Value(5).is<SimpleJSON::Type::Integer>(), true);
}

TEST(VAlue,Number) {
	ASSERT_EQ(SimpleJSON::Value(5.0),SimpleJSON::Type::Number(5));
	ASSERT_EQ(SimpleJSON::Type::Number(5.0),5.0);
	ASSERT_EQ(SimpleJSON::Type::Number(5),5.0);
	SimpleJSON::Value val(4.0);
	val=5;
	ASSERT_EQ(val,5);
	ASSERT_EQ(SimpleJSON::Value(5).is<SimpleJSON::Type::Integer>(), true);
}

TEST(T,String) {
	ASSERT_EQ(SimpleJSON::Value(std::string("hello")),SimpleJSON::Type::String("hello"));
}

TEST(T,Array) {
	ASSERT_EQ(SimpleJSON::Value(std::string("hello")),SimpleJSON::Type::String("hello"));
}

/*
TEST(T,Array) {
	auto array= SimpleJSON::Type::Array({1,2,3,4,5});
	ASSERT_EQ(array[0], 1);
	ASSERT_EQ(array[1], 2);
}
*/