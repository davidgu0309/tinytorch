#include "../framework/test.hpp"
#include "../../include/tensor_operation/power.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace power_tests {

Power<double> power_op;

Tensor<double> forward(const std::vector<Tensor<double>> operands){
    return power_op(operands);
}

TestSuite<forward> forwardUnitTests(){
    TestSuite<forward> forward_tests("Power operator tests");
    forward_tests.addTest(ComparativeUnitTest<forward>(std::vector<Tensor<double>>({Tensor<double>(2), Tensor<double>(10)}), Tensor<double>(1024)));
    forward_tests.addTest(ComparativeUnitTest<forward>(std::vector<Tensor<double>>({constant<double>({5}, 1), constant<double>({5}, 1)}), constant<double>({5}, 1)));
    return forward_tests;
}

Tensor<double> backward(const size_t input_idx, const std::vector<Tensor<double>> operands){
    return power_op.backward(input_idx, operands);
}

TestSuite<backward> backwardUnitTests(){
    TestSuite<backward> backward_tests("Power backward tests");
    backward_tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<double>>({Tensor<double>(2), Tensor<double>(2)})}, Tensor<double>(4)));
    backward_tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<double>>({Tensor<double>(3), Tensor<double>(3)})}, Tensor<double>(27)));
    backward_tests.addTest(ComparativeUnitTest<backward>({1, std::vector<Tensor<double>>({Tensor<double>(exp(1)), Tensor<double>(3)})}, Tensor<double>(exp(1) * exp(1) * exp(1))));
    return backward_tests;
}

void runUnitTests() {

    forwardUnitTests().run();
    backwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

