#include <SimpleJSON.h>

#include <gtest/gtest.h>

TEST(Simple,String) {
	ASSERT_EQ(SimpleJSON::parse("\"Ahoj\""), SimpleJSON::Type::String("Ahoj"));
	ASSERT_EQ(SimpleJSON::parse("  \"Ahoj\"  "), SimpleJSON::Type::String("Ahoj"));
	//ASSERT_EQ(SimpleJSON::parse("\"Ahoj\\\"\"").as<std::string>(), "Ahoj\"");
}

TEST(Simple,Bool) {
	ASSERT_EQ(SimpleJSON::parse("true"), SimpleJSON::Type::Boolean(true));
	ASSERT_EQ(SimpleJSON::parse("false"), SimpleJSON::Type::Boolean(false));
	ASSERT_EQ(SimpleJSON::parse("  false  "), SimpleJSON::Type::Boolean(false));
}

TEST(Object,Strings) {
	ASSERT_EQ(SimpleJSON::parse("{\"a\": \"b\"}").as<SimpleJSON::Type::Object>()["a"], std::string("b"));
	ASSERT_EQ(SimpleJSON::parse("{\"a\": \"b\",\"c\": \"d\"}").as<SimpleJSON::Type::Object>()["c"], std::string("d"));
	ASSERT_EQ(SimpleJSON::parse("{\"a\":\"b\",\"c\":\"d\"}").as<SimpleJSON::Type::Object>()["c"], std::string("d"));
	ASSERT_EQ(SimpleJSON::parse("{\"a\":\"b\" , \"c\":\"d\"}").as<SimpleJSON::Type::Object>()["c"], std::string("d"));
}

TEST(Simple,Number) {
	ASSERT_EQ(SimpleJSON::parse("10.0"), SimpleJSON::Type::Number(10));
	ASSERT_EQ(SimpleJSON::parse(" 10.0"), SimpleJSON::Type::Number(10));
	ASSERT_EQ(SimpleJSON::parse("10.0 "), SimpleJSON::Type::Number(10));
}

TEST(Simple,Integer) {
	ASSERT_EQ(SimpleJSON::parse("10"), SimpleJSON::Type::Integer(10));
	ASSERT_EQ(SimpleJSON::parse(" 10"), SimpleJSON::Type::Integer(10));
	ASSERT_EQ(SimpleJSON::parse("10 "), SimpleJSON::Type::Integer(10));
}

TEST(Array,Strings) {
	ASSERT_EQ(SimpleJSON::parse(" [\"a\",\"b\"] ").as<SimpleJSON::Type::Array>().size(), 2);
	//ASSERT_EQ(SimpleJSON::parse(" [ \"a\" , \"b\" ] ").as<SimpleJSON::Type::Array>()[0], std::string("a"));
}

TEST(Simple,StringsAndBools) {
	ASSERT_EQ(SimpleJSON::parse("{\"a\": \"b\",\"c\": false}").as<SimpleJSON::Type::Object>()["c"], false);
}