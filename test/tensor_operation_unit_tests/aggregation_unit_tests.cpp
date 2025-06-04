#include "../framework/test.hpp"
#include "../../include/tensor_operation/aggregation.hpp"
#include "../../tensor/test/test_tensors.hpp"

using namespace tensor;
using namespace test;
using namespace tinytorch;

namespace tensor_aggregation_tests {

Sum<int> sum_op;

Tensor<int> sum(const std::vector<Tensor<int>> operands){
    return sum_op(operands);
}

TestSuite<sum> sumUnitTests(){
    TestSuite<sum> tests("Sum operator tests");
    tests.addTest(ComparativeUnitTest<sum>(std::vector<Tensor<int>>({scalar_10}), scalar_10));
    tests.addTest(ComparativeUnitTest<sum>(std::vector<Tensor<int>>({iota_3x3}), Tensor<int>(45)));
    return tests;
}

Tensor<int> sumBackward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return sum_op.backward(input_idx, operands);
}

TestSuite<sumBackward> sumBackwardUnitTests(){
    TestSuite<sumBackward> tests("Sum backward tests");
    tests.addTest(ComparativeUnitTest<sumBackward>({0, std::vector<Tensor<int>>({scalar_10})}, scalar_1));
    tests.addTest(ComparativeUnitTest<sumBackward>({0, std::vector<Tensor<int>>({ones_5})}, ones_5));
    tests.addTest(ComparativeUnitTest<sumBackward>({0, std::vector<Tensor<int>>({iota_5})}, ones_5));
    return tests;
}

Product<int> product_op;

Tensor<int> product(const std::vector<Tensor<int>> operands){
    return product_op(operands);
}

TestSuite<product> productUnitTests(){
    TestSuite<product> tests("Product operator tests");
    tests.addTest(ComparativeUnitTest<product>(std::vector<Tensor<int>>({scalar_10}), scalar_10));
    tests.addTest(ComparativeUnitTest<product>(std::vector<Tensor<int>>({ones_5}), Tensor<int>(1)));
    tests.addTest(ComparativeUnitTest<product>(std::vector<Tensor<int>>({iota_5}), Tensor<int>(120)));
    return tests;
}

Tensor<int> productBackward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return product_op.backward(input_idx, operands);
}

TestSuite<productBackward> productBackwardUnitTests(){
    TestSuite<productBackward> tests("Product backward tests");
    tests.addTest(ComparativeUnitTest<productBackward>({0, std::vector<Tensor<int>>({scalar_10})}, scalar_1));
    tests.addTest(ComparativeUnitTest<productBackward>({0, std::vector<Tensor<int>>({ones_5})}, ones_5));
    return tests;
}

ArithmeticMean<int> arithmetic_mean_op;

Tensor<int> arithmeticMean(const std::vector<Tensor<int>> operands){
    return arithmetic_mean_op(operands);
}

TestSuite<arithmeticMean> arithmeticMeanUnitTests(){
    TestSuite<arithmeticMean> tests("Arithmetic mean operator tests");
    tests.addTest(ComparativeUnitTest<arithmeticMean>(std::vector<Tensor<int>>({scalar_10}), scalar_10));
    tests.addTest(ComparativeUnitTest<arithmeticMean>(std::vector<Tensor<int>>({iota_3x3}), Tensor<int>(5)));
    return tests;
}

Tensor<int> arithmeticMeanBackward(const size_t input_idx, const std::vector<Tensor<int>> operands){
    return arithmetic_mean_op.backward(input_idx, operands);
}

TestSuite<arithmeticMeanBackward> arithmeticMeanBackwardUnitTests(){
    TestSuite<arithmeticMeanBackward> tests("Arithmetic mean backward tests");
    tests.addTest(ComparativeUnitTest<arithmeticMeanBackward>({0, std::vector<Tensor<int>>({scalar_10})}, scalar_1));
    tests.addTest(ComparativeUnitTest<arithmeticMeanBackward>({0, std::vector<Tensor<int>>({ones_5})}, zeros<int>({5})));
    return tests;
}

void runUnitTests() {

    sumUnitTests().run();
    sumBackwardUnitTests().run();

    productUnitTests().run();
    productBackwardUnitTests().run();

    arithmeticMeanUnitTests().run();
    arithmeticMeanBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

