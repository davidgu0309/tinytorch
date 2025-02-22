#include "../framework/test.hpp"
#include "../../tensor_operation/include/addition.hpp"

using namespace test;
using namespace tinytorch;

// TO DO: make a file in tensor for this, maybe with some macros too
Tensor<int> zeros_3x4x5 = zeros<int>({3, 4, 5});
Tensor<int> ones_5 = ones<int>({5});
Tensor<int> twos_5 = constant<int>({5}, 2);
Tensor<int> threes_5 = constant<int>({5}, 3);
Tensor<int> ones_10 = ones<int>({10});
Tensor<int> ones_3x3 = ones<int>({3, 3});
Tensor<int> ones_3x2x4 = ones<int>({3, 2, 4});
Tensor<int> t1 = Tensor<int>({3, 2, 4}, std::vector<int>(24, -1));
Tensor<int> t2 = Tensor<int>({10}, std::vector<int>(10, -1));
Tensor<int> t3 = Tensor<int>({3, 3}, std::vector<int>(9, -1));
Tensor<int> threes_3x3 = constant<int>({3, 3}, 3);
Tensor<int> iota_5 = iota<int>({5});
Tensor<int> scalar_10 = Tensor<int>(10);
Tensor<int> scalar_15 = Tensor<int>(15);
Tensor<int> scalar_55 = Tensor<int>(55);
Tensor<int> iota_3x3 = iota<int>({3,3});
Tensor<int> iota_2x2x3 = iota<int>({2,2,3});
Tensor<int> iota_3x3_squared = Tensor<int>({3,3}, std::vector<int>({30,36,42,66,81,96,102,126,150}));
Tensor<int> iota_3D_times_2D_result = Tensor<int>({2,2,3}, std::vector<int>{30,36,42,66,81,96,102,126,150,138,171,204});

TensorAddition<int> tensor_addition_operation;

Tensor<int> addition(const std::vector<Tensor<int>> operands){
    return tensor_addition_operation(operands);
}

TestSuite<addition> additionUnitTests(){
    TestSuite<addition> addition_tests;
    std::vector<Tensor<int>> operands({ones_5, twos_5});
    UnitTest<addition> unit_test(operands, threes_5);
    addition_tests.addTest(unit_test);
    return addition_tests;
}

Tensor<int> backward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return tensor_addition_operation.backward(input_idx, operands);
}

TestSuite<backward> backwardUnitTests(){
    TestSuite<backward> backward_tests;
    return backward_tests;
}

void tensorAdditionUnitTests() {

    
    additionUnitTests().run();
    backwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

