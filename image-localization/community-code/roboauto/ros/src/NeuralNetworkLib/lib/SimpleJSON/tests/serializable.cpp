#include <gtest/gtest.h>

#include <SimpleJSON/SerializableObject.h>

class A : public SimpleJSON::SerializableObject {

	public:
		virtual SimpleJSON::Type::Object serialize() const override {
			return {{"class", "A"}};
		}

};

TEST(Factory,One) {
	ASSERT_EQ(A().serialize()["class"], std::string("A"));
	ASSERT_EQ(SimpleJSON::Value(A()).as<SimpleJSON::Type::Object>()["class"], std::string("A"));
}