#include <gtest/gtest.h>

#include <SimpleJSON/SerializableObject.h>
#include <SimpleJSON/Factory.h>

class A : public SimpleJSON::SerializableObject {

	public:
		A(int a_) : test(a_) {

		}
		virtual SimpleJSON::Type::Object serialize() const override {
			return {{"class", "A"},{"test", test}};
		}

		static std::unique_ptr<A> deserialize(const SimpleJSON::Type::Object& obj) {
			return std::unique_ptr<A>(new A(obj["test"].as<int>()));
		}
		int test = 8;
};

typedef SimpleJSON::Factory<A> Factory;

TEST(Factory,One) {
	Factory::registerCreator("A",A::deserialize);
	A object = A(7);
	std::string serialized = A(7).serialize().serialize();
	ASSERT_EQ(Factory::deserialize(object.serialize())->test,7);
	ASSERT_EQ(Factory::deserialize(serialized)->test,7);
}