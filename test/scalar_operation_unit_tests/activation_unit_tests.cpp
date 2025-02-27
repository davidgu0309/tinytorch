#include "../framework/test.hpp"
#include "../../include/scalar_operation/activation.hpp"

using namespace test;
using namespace tinytorch;

double INF = std::numeric_limits<double>::infinity();

namespace activation_tests {

ScalarReLU<double> relu_op;

double relu(const std::vector<double> operands){
    return relu_op(operands);
}

TestSuite<relu> reluUnitTests(){
    TestSuite<relu> relu_tests("ReLU operator tests");
    relu_tests.addTest(ComparativeUnitTest<relu>(std::vector<double>({-3.7}), 0.0));
    relu_tests.addTest(ComparativeUnitTest<relu>(std::vector<double>({0.0}), 0.0));
    relu_tests.addTest(ComparativeUnitTest<relu>(std::vector<double>({1e9}), 1e9));
    return relu_tests;
}

double reluBackward(const size_t input_idx, const std::vector<double> operands){
    return relu_op.backward(input_idx, operands);
}

TestSuite<reluBackward> reluBackwardUnitTests(){
    TestSuite<reluBackward> relu_backward_tests("ReLU backward tests");
    relu_backward_tests.addTest(ComparativeUnitTest<reluBackward>({0, std::vector<double>({-3.7})}, 0.0));
    relu_backward_tests.addTest(ComparativeUnitTest<reluBackward>({0, std::vector<double>({0.0})}, 0.0));
    relu_backward_tests.addTest(ComparativeUnitTest<reluBackward>({0, std::vector<double>({1e9})}, 1.0));
    return relu_backward_tests;
}

ScalarSigmoid<double> sigmoid_op;

double sigmoid(const std::vector<double> operands){
    return sigmoid_op(operands);
}

TestSuite<sigmoid> sigmoidUnitTests(){
    TestSuite<sigmoid> sigmoid_tests("Sigmoid operator tests");
    sigmoid_tests.addTest(ComparativeUnitTest<sigmoid>(std::vector<double>({0.0}), 0.5));
    sigmoid_tests.addTest(ComparativeUnitTest<sigmoid>(std::vector<double>({INF}), 1.0));
    sigmoid_tests.addTest(ComparativeUnitTest<sigmoid>(std::vector<double>({-INF}), 0.0));
    return sigmoid_tests;
}

double sigmoidBackward(const size_t input_idx, const std::vector<double> operands){
    return sigmoid_op.backward(input_idx, operands);
}

TestSuite<sigmoidBackward> sigmoidBackwardUnitTests(){
    TestSuite<sigmoidBackward> sigmoid_backward_tests("Sigmoid backward tests");
    sigmoid_backward_tests.addTest(ComparativeUnitTest<sigmoidBackward>({0, std::vector<double>({0.0})}, 0.25));
    sigmoid_backward_tests.addTest(ComparativeUnitTest<sigmoidBackward>({0, std::vector<double>({-INF})}, 0.0));
    sigmoid_backward_tests.addTest(ComparativeUnitTest<sigmoidBackward>({0, std::vector<double>({INF})}, 0.0));
    return sigmoid_backward_tests;
}

void runUnitTests() {

    reluUnitTests().run();
    reluBackwardUnitTests().run();

    sigmoidUnitTests().run();
    sigmoidBackwardUnitTests().run();

    // TODO: sigmoid tests (probably should implement eps-comparisons for floats in testing framework)

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

