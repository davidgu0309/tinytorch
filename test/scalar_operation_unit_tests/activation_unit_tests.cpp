#include "../framework/test.hpp"
#include "../../include/scalar_operation/activation.hpp"

using namespace test;
using namespace tinytorch;

namespace activation_tests {

ScalarReLU<double> relu_op;

double relu(const std::vector<double> operands){
    return relu_op(operands);
}

TestSuite<relu> reluUnitTests(){
    TestSuite<relu> relu_tests;
    relu_tests.addTest(ComparativeUnitTest<relu>(std::vector<double>({-3.7}), 0.0));
    relu_tests.addTest(ComparativeUnitTest<relu>(std::vector<double>({0.0}), 0.0));
    relu_tests.addTest(ComparativeUnitTest<relu>(std::vector<double>({1e9}), 1e9));
    return relu_tests;
}

double reluBackward(const size_t input_idx, const std::vector<double> operands){
    return relu_op.backward(input_idx, operands);
}

TestSuite<reluBackward> reluBackwardUnitTests(){
    TestSuite<reluBackward> relu_backward_tests;
    relu_backward_tests.addTest(ComparativeUnitTest<reluBackward>({0, std::vector<double>({-3.7})}, 0.0));
    relu_backward_tests.addTest(ComparativeUnitTest<reluBackward>({0, std::vector<double>({0.0})}, 0.0));
    relu_backward_tests.addTest(ComparativeUnitTest<reluBackward>({0, std::vector<double>({1e9})}, 1.0));
    return relu_backward_tests;
}

void runUnitTests() {

    std::cout << "----- ReLU operator tests -----" << std::endl;
    reluUnitTests().run();
    std::cout << "----- ReLU backward tests -----" << std::endl;
    reluBackwardUnitTests().run();

    // TODO: sigmoid tests (probably should implement eps-comparisons for floats in testing framework)

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

