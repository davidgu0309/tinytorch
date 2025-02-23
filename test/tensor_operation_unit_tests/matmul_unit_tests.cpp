#include "../framework/test.hpp"
#include "../../tensor_operation/include/matmul.hpp"
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
    TestSuite<operation> matmul_tests;
    matmul_tests.addTest(UnitTest<operation>(std::vector<Tensor<int>>({ones_5, twos_5}), scalar_10));
    return matmul_tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return matmul_op.backward(input_idx, operands);
}

TestSuite<backward> backwardUnitTests(){
    TestSuite<backward> backward_tests;
    backward_tests.addTest(UnitTest<backward>({0, std::vector<Tensor<int>>({ones_5, twos_5})}, twos_5));
    backward_tests.addTest(UnitTest<backward>({1, std::vector<Tensor<int>>({ones_5, twos_5})}, ones_5));
    return backward_tests;
}

void runUnitTests() {

    std::cout << "----- Matmul operator tests -----" << std::endl;
    matmulUnitTests().run();
    std::cout << "----- Matmul backward tests -----" << std::endl;
    backwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace matmultest

