#include "../include/functional.hpp"
#include "../include/tensor.hpp"

#include <iostream>

namespace tinytorch {

    template<typename T>
    void consoleLog(const T& result, const T& correct){
        bool passed = result == correct;
        std::cout << (passed ? "Passed" : "Failed") << std::endl;
        if(!passed){
            std::cout << "Result:" << std::endl;
            std::cout << result << std::endl;
            std::cout << "Correct:" << std::endl;
            std::cout << result << std::endl;
        }
    }

    // struct Test{}; // Maybe think about this

    template<typename T, typename U, U (*unaryOp)(const T&)>
    struct UnaryOpTest {
        T& t;
        U& correct;
        bool run() const {
            U result = unaryOp(t);
            bool test_passed = result == correct;
            consoleLog<U>(result, correct);
            return test_passed;
        }
    };

    template<typename T, typename U, typename V, V (*binaryOp)(const T&, const U&)>
    struct BinaryOpTest {
        T& t1;
        U& t2;
        V& correct;
        bool run() const {
            V result = binaryOp(t1, t2);
            bool test_passed = result == correct;
            consoleLog<V>(result, correct);
            return test_passed;
        }
    };

    template<typename T, typename U, U (*unaryOp)(const T&)>
    using UnaryOpTestSuite = std::vector<UnaryOpTest<T, U, unaryOp> >;

    template<typename T, typename U, T (*unaryOp)(const T&)>
    void runUnaryOpTestSuite(UnaryOpTestSuite<T, U, unaryOp> tests){
        for(const UnaryOpTest<T, U, unaryOp>& test : tests){
            test.run();
        }
    }

    template<typename T, typename U, typename V, V (*binaryOp)(const T&, const U&)>
    using BinaryOpTestSuite = std::vector<BinaryOpTest<T, U, V, binaryOp> >;

    template<typename T, typename U, typename V, V (*binaryOp)(const T&, const U&)>
    void runBinaryOpTestSuite(BinaryOpTestSuite<T, U, V, binaryOp> tests){
        for(const BinaryOpTest<T, U, V, binaryOp>& test : tests){
            test.run();
        }
    }

    Tensor<int> negInt(const Tensor<int>& a){
        return neg<int>(a);
    }

    void functionalUnitTests() {

        Tensor<int> t1 = Tensor<int>(std::vector<int>(24, -1), {3, 2, 4});
        Tensor<int> t2 = Tensor<int>(std::vector<int>(10, -1), {10});
        Tensor<int> t3 = Tensor<int>(std::vector<int>(9, -1), {3, 3});
        
        UnaryOpTestSuite<Tensor<int>, Tensor<int>, negInt> neg_tests = {
            {zeros<int>({3, 4, 5}), zeros<int>({3, 4, 5})},
            {ones<int>({3, 2, 4}), t1},
            {ones<int>({10}), t2},
            {ones<int>({3, 3}), t3},
        };

        std::cout << "Neg Tests" << std::endl;
        runUnaryOpTestSuite<Tensor<int>, Tensor<int>, negInt>(neg_tests);

    }

}