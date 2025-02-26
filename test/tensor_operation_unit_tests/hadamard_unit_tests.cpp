#include "../framework/test.hpp"
#include "../../include/tensor_operation/hadamard.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace hadamard_tests {

Hadamard<int> hadamard_op;

Tensor<int> hadamard(const std::vector<Tensor<int>> operands){
    return hadamard_op(operands);
}

TestSuite<hadamard> hadamardUnitTests(){
    TestSuite<hadamard> hadamard_tests;
    hadamard_tests.addTest(UnitTest<hadamard>(std::vector<Tensor<int>>({scalar_10}), scalar_10));
    hadamard_tests.addTest(UnitTest<hadamard>(std::vector<Tensor<int>>({ones_5, twos_5}), twos_5));
    return hadamard_tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return hadamard_op.backward(input_idx, operands);
}

TestSuite<backward> backwardUnitTests(){
    TestSuite<backward> backward_tests;
    backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({scalar_10, scalar_15})}, scalar_15));
    backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({iota_5, ones_5})}, id_5x5));
    // TODO: build result tensor for this one (zeros with iota on the diag)
    // backward_tests.addTest(UnitTest<backward>({1, std::vector<Tensor<int>>({iota_5, ones_5})}, id_5x5));

    /*
        _ _ _ _
        x11 x12 x13     y11 y12 y13     x11*y11 x12*y12 x13*y13
        x21 x22 x23  *  y21 y22 y23  =  x21*y21 x22*y22 x23*y23
        x31 x32 x33     y31 y32 y33     x31*y31 x32*y32 x33*y33
    */
    // backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({ones_3x3, ones_3x3})}, id_5x5));
    return backward_tests;
}

void runUnitTests() {

    std::cout << "----- Hadamard operator tests -----" << std::endl;
    hadamardUnitTests().run();
    std::cout << "----- Hadamard backward tests -----" << std::endl;
    backwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

