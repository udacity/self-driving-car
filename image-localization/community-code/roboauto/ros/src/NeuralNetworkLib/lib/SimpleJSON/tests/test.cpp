#include <SimpleJSON.h>
#include <SimpleJSON/SerializableObject.h>

#include <string>
#include <iostream>

class A : public SimpleJSON::SerializableObject {
	public:
		virtual SimpleJSON::Type::Object serialize() const override {
				return {{"class","A"}};
		}
};

class B : public SimpleJSON::SerializableObject {

	public:
		virtual SimpleJSON::Type::Object serialize() const override {
			return {
				{"class","B"},
				{"X",A()}};
		}
};

int main() {
	std::cout << SimpleJSON::Value(5) << std::endl;
	std::cout << A() << std::endl;
	std::cout << B() << std::endl;
	std::cout << SimpleJSON::Value(A());
}