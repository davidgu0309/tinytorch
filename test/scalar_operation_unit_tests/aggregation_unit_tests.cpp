#include "../framework/test.hpp"
#include "../../include/scalar_operation/aggregation.hpp"

using namespace test;
using namespace tinytorch;

namespace aggregation_tests {

ScalarAddition<double> add_op;

double add(const std::vector<double> operands){
    return add_op(operands);
}

TestSuite<add> addUnitTests(){
    TestSuite<add> tests("Addition operator tests");
    tests.addTest(ComparativeUnitTest<add>(std::vector<double>({-3.7, 4.3, 1.0, -1.6}), 0.0, epsilon_comparison(1e-9)));
    return tests;
}

double addBackward(const size_t input_idx, const std::vector<double> operands){
    return add_op.backward(input_idx, operands);
}

TestSuite<addBackward> addBackwardUnitTests(){
    TestSuite<addBackward> tests("Addition backward tests");
    tests.addTest(ComparativeUnitTest<addBackward>({0, std::vector<double>({-3.7})}, 1.0));
    tests.addTest(ComparativeUnitTest<addBackward>({0, std::vector<double>({-3.7, 1.0})}, 1.0));
    tests.addTest(ComparativeUnitTest<addBackward>({1, std::vector<double>({-3.7, 1.0})}, 1.0));
    return tests;
}

ScalarMultiplication<double> mul_op;

double mul(const std::vector<double> operands){
    return mul_op(operands);
}

TestSuite<mul> mulUnitTests(){
    TestSuite<mul> tests("Multiplication operator tests");
    tests.addTest(ComparativeUnitTest<mul>(std::vector<double>({1.0, 2.0, 3.0, 4.1}), 24.6, epsilon_comparison(1e-9)));
    return tests;
}

double mulBackward(const size_t input_idx, const std::vector<double> operands){
    return mul_op.backward(input_idx, operands);
}

TestSuite<mulBackward> mulBackwardUnitTests(){
    TestSuite<mulBackward> tests("Multiplication backward tests");
    tests.addTest(ComparativeUnitTest<mulBackward>({0, std::vector<double>({-3.7})}, 1.0));
    tests.addTest(ComparativeUnitTest<mulBackward>({0, std::vector<double>({-3.7, 1.0, 2.0})}, 2.0));
    tests.addTest(ComparativeUnitTest<mulBackward>({1, std::vector<double>({-3.7, 1.0, 2.0})}, -7.4));
    tests.addTest(ComparativeUnitTest<mulBackward>({2, std::vector<double>({-3.7, 1.0, 2.0})}, -3.7));
    return tests;
}

void runUnitTests() {

    addUnitTests().run();
    addBackwardUnitTests().run();

    mulUnitTests().run();
    mulBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

