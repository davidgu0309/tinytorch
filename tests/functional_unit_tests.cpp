#include "../include/functional.hpp"
#include "../include/tensor.hpp"

#include <iostream>

namespace tinytorch {

    // struct Test{}; // Maybe think about this

    template<typename T, typename U>
    struct UnaryOpTest {
        Tensor<T>& t;
        Tensor<U>& result;
    };

    template<typename T, typename U, U (*unaryOp)(T)>
    bool unaryOpTest(const UnaryOpTest<T, U>& test_data) {
        Tensor<int> result = applyUnaryOp<T, U, unaryOp>(test_data.t);
        std::cout << result << std::endl;
        bool test_passed = result == test_data.result;
        std::cout << (test_passed ? "Passed" : "Failed") << std::endl;
        return test_passed;
    }

    template<typename T, typename U, typename V>
    struct BinaryOpTest {
        Tensor<T>& t1;
        Tensor<U>& t2;
        Tensor<V>& result;
    };

    template<typename T, typename U, typename V, V (*binaryOp)(T, U)>
    bool binaryOpTest(const BinaryOpTest<T, U, V>& test_data) {
        Tensor<int> result = applyBinaryOp<T, U, V, binaryOp>(test_data.t1, test_data.t2);
        bool test_passed = result == test_data.result;
        std::cout << (test_passed ? "Passed" : "Failed") << std::endl;
        return test_passed;
    }

    template<typename T, typename U>
    using UnaryOpTestSuite = std::vector<UnaryOpTest<T, U> >;

    template<typename T, typename U, U (*unaryOp)(T)>
    void runUnaryOpTestSuite(UnaryOpTestSuite<T, U> tests){
        for(const UnaryOpTest<T, U>& test : tests){
            unaryOpTest<T, U, unaryOp>(test);
        }
    }

    template<typename T, typename U, typename V>
    using BinaryOpTestSuite = std::vector<BinaryOpTest<T, U, V> >;

    template<typename T, typename U, typename V, V (*binaryOp)(T, U)>
    void runBinaryOpTestSuite(BinaryOpTestSuite<T, U, V> tests){
        for(const BinaryOpTest<T, U, V>& test : tests){
            binaryOpTest<T, U, V, binaryOp>(test);
        }
    }

    int scalarNegInt(int a){
        return scalarNeg<int>(a);
    }

    void functionalUnitTests() {

        Tensor<int> t1 = Tensor<int>(std::vector<int>(210, -1), {7, 6, 5});
        Tensor<int> t2 = Tensor<int>(std::vector<int>(10, -1), {10});
        Tensor<int> t3 = Tensor<int>(std::vector<int>(9, -1), {3, 3});
        


        UnaryOpTestSuite<int, int> neg_tests = {
            {zeros<int>({3, 4, 5}), zeros<int>({3, 4, 5})},
            {ones<int>({7, 6, 5}), t1},
            {ones<int>({10}), t2},
            {ones<int>({3, 3}), t3},
        };

        std::cout << "Neg Tests" << std::endl;
        runUnaryOpTestSuite<int, int, scalarNegInt>(neg_tests);

    }

}