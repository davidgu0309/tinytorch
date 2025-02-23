#include "../framework/test.hpp"
#include "../../tensor_operation/include/addition.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace addition {

Addition<int> addition_op;

Tensor<int> addition(const std::vector<Tensor<int>> operands){
    return addition_op(operands);
}

TestSuite<addition> additionUnitTests(){
    TestSuite<addition> addition_tests;
    addition_tests.addTest(UnitTest<addition>(std::vector<Tensor<int>>({ones_5, twos_5}), threes_5));
    return addition_tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return addition_op.backward(input_idx, operands);
}

TestSuite<backward> backwardUnitTests(){
    TestSuite<backward> backward_tests;
    backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({scalar_10, scalar_15})}, scalar_1));
    backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({ones_5, ones_5})}, id_5x5));
    return backward_tests;
}

void runUnitTests() {

    std::cout << "----- Addition operator tests -----" << std::endl;
    additionUnitTests().run();
    std::cout << "----- Addition backward tests -----" << std::endl;
    backwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

