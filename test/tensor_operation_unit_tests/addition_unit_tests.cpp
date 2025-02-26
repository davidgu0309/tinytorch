#include "../framework/test.hpp"
#include "../../include/tensor_operation/addition.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace addition_tests {

Addition<int> addition_op;

Tensor<int> addition(const std::vector<Tensor<int>> operands){
    return addition_op(operands);
}

TestSuite<addition> additionUnitTests(){
    TestSuite<addition> addition_tests;
    addition_tests.addTest(UnitTest<addition>(std::vector<Tensor<int>>({scalar_10}), scalar_10));
    addition_tests.addTest(UnitTest<addition>(std::vector<Tensor<int>>({scalar_15, scalar_15, scalar_10, scalar_15}), scalar_55));
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

    /*
        _ _ _ _
        x11 x12 x13     y11 y12 y13     x11+y11 x12+y12 x13+y13
        x21 x22 x23  +  y21 y22 y23  =  x21+y21 x22+y22 x23+y23
        x31 x32 x33     y31 y32 y33     x31+y31 x32+y32 x33+y33
    */
    // backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({ones_3x3, ones_3x3})}, id_5x5));
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

