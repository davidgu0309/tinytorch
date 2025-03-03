#include "../framework/test.hpp"
#include "../../include/tensor_operation/loss.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace tensor_loss_tests {

Lp<int> lp_op;

Tensor<int> lp(const std::vector<Tensor<int>> operands){
    return lp_op(operands);
}

TestSuite<lp> lpUnitTests(){
    TestSuite<lp> tests("Lp loss operator tests");
    tests.addTest(ComparativeUnitTest<lp>(std::vector<Tensor<int>>({scalar_10, scalar_10}), zeros<int>({})));
    tests.addTest(ComparativeUnitTest<lp>(std::vector<Tensor<int>>({scalar_10, scalar_15}), constant({}, 25)));
    tests.addTest(ComparativeUnitTest<lp>(std::vector<Tensor<int>>({iota_2x2x3, iota_2x2x3}), zeros<int>({2, 2, 3})));
    return tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return lp_op.backward(input_idx, operands);
}

TestSuite<backward> lpBackwardUnitTests(){
    TestSuite<backward> tests("ReLU backward tests");
    tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<int>>({scalar_10, scalar_10})}, zeros<int>({})));
    tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<int>>({scalar_10, scalar_15})}, constant({}, -10)));
    tests.addTest(ComparativeUnitTest<backward>({1, std::vector<Tensor<int>>({scalar_10, scalar_10})}, zeros<int>({})));
    tests.addTest(ComparativeUnitTest<backward>({1, std::vector<Tensor<int>>({scalar_10, scalar_15})}, scalar_10));
    return tests;
}

void runUnitTests() {

    lpUnitTests().run();
    lpBackwardUnitTests().run();

    // sigmoidUnitTests().run();
    // sigmoidBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

