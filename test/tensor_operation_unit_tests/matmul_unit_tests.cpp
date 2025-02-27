#include "../framework/test.hpp"
#include "../../include/tensor_operation/matmul.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace matmul_tests{

Matmul<int> matmul_op;

Tensor<int> operation(const std::vector<Tensor<int>> operands){
    return matmul_op(operands);
}

TestSuite<operation> matmulUnitTests(){
    TestSuite<operation> matmul_tests("Matmul operator tests");
    matmul_tests.addTest(ComparativeUnitTest<operation>(std::vector<Tensor<int>>({ones_5, twos_5}), scalar_10));
    return matmul_tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return matmul_op.backward(input_idx, operands);
}

TestSuite<backward> backwardUnitTests(){
    TestSuite<backward> backward_tests("Matmul backward tests");
    /*
        x1 x2 x3 x4 x5 @ y1 y2 y3 y4 y5 = sum_i xi * yi
    */
    backward_tests.addTest(ComparativeUnitTest<backward>({0, std::vector<Tensor<int>>({ones_5, twos_5})}, twos_5));
    backward_tests.addTest(ComparativeUnitTest<backward>({1, std::vector<Tensor<int>>({ones_5, twos_5})}, ones_5));
    /*

        x11 x12 x13     y11 y12 y13     _ _ _
        x21 x22 x23  @  y21 y22 y23  =  _ _ _
        x31 x32 x33     y31 y32 y33     _ _ _
    */
    Tensor<int> ones_3x3x3x3 = ones<int>({3, 3, 3, 3});
    // std::cout << backward(1, std::vector<Tensor<int>>({ones_3x3, ones_3x3}));
    // backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({ones_3x3, ones_3x3})}, ones_3x3x3x3));
    return backward_tests;
}

void runUnitTests() {

    matmulUnitTests().run();
    backwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace matmultest

