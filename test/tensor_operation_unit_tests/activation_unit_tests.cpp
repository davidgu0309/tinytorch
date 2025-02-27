#include "../framework/test.hpp"
#include "../../include/tensor_operation/activation.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace tensor_activation_tests {

ReLU<int> relu_op;

Tensor<int> relu(const std::vector<Tensor<int>> operands){
    return relu_op(operands);
}

TestSuite<relu> reluUnitTests(){
    TestSuite<relu> tests("ReLU operator tests");
    tests.addTest(ComparativeUnitTest<relu>(std::vector<Tensor<int>>({scalar_10}), scalar_10));
    tests.addTest(ComparativeUnitTest<relu>(std::vector<Tensor<int>>({Tensor<int>(-4)}), zeros<int>({})));
    return tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return relu_op.backward(input_idx, operands);
}

TestSuite<backward> reluBackwardUnitTests(){
    TestSuite<backward> tests("ReLU backward tests");
    tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<int>>({scalar_10})}, scalar_1));
    tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<int>>({ones_5})}, id_5x5));
    return tests;
}

void runUnitTests() {

    reluUnitTests().run();
    reluBackwardUnitTests().run();

    // sigmoidUnitTests().run();
    // sigmoidBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

