#include "../tensor/include/functional.hpp"
#include "../tensor/include/tensor.hpp"

#include <iostream>
#include <numeric>

using namespace tensor;
using namespace tinytorch;

namespace test{
    
    template<typename T>
    void consoleLog(const T& result, const T& correct){
        bool passed = result == correct;
        std::cout << (passed ? "Passed" : "Failed") << std::endl;
        if(!passed){
            std::cout << "Result:" << std::endl;
            std::cout << result << std::endl;
            std::cout << "Correct:" << std::endl;
            std::cout << correct << std::endl;
        }
    }

    // struct Test{}; // Maybe think about this

    template<typename T, typename U, U (*unaryOp)(const T&)>
    struct UnaryOpTest {
        T operand;
        U correct;
        bool run() const {
            U result = unaryOp(operand);
            bool test_passed = result == correct;
            // std::cout << operand << std::endl;
            consoleLog<U>(result, correct);
            return test_passed;
        }
    };

    template<typename T, typename U, typename V, V (*binaryOp)(const T&, const U&)>
    struct BinaryOpTest {
        T operand_1;
        U operand_2;
        V correct;
        bool run() const {
            V result = binaryOp(operand_1, operand_2);
            bool test_passed = result == correct;
            // std::cout << operand_1 << std::endl;
            // std::cout << operand_2 << std::endl;
            consoleLog<V>(result, correct);
            return test_passed;
        }
    };

    // TO DO: implement << for tests

    template<typename T, typename U, U (*unaryOp)(const T&)>
    using UnaryOpTestSuite = std::vector<UnaryOpTest<T, U, unaryOp> >;

    template<typename T, typename U, T (*unaryOp)(const T&)>
    void runUnaryOpTestSuite(UnaryOpTestSuite<T, U, unaryOp> tests){
        for(const UnaryOpTest<T, U, unaryOp>& test : tests){
            test.run();
        }
        std::cout << std::endl;
    }

    template<typename T, typename U, typename V, V (*binaryOp)(const T&, const U&)>
    using BinaryOpTestSuite = std::vector<BinaryOpTest<T, U, V, binaryOp> >;

    template<typename T, typename U, typename V, V (*binaryOp)(const T&, const U&)>
    void runBinaryOpTestSuite(BinaryOpTestSuite<T, U, V, binaryOp> tests){
        for(const BinaryOpTest<T, U, V, binaryOp>& test : tests){
            test.run();
        }
        std::cout << std::endl;
    }

    Tensor<int> negInt(const Tensor<int>& a){
        return neg<int>(a);
    }

    Tensor<int> addInt(const Tensor<int>& a, const Tensor<int>& b){
        return add<int>(a, b);
    }

    Tensor<int> matmulInt(const Tensor<int>& a, const Tensor<int>& b){
        return matmul<int>(a, b);
    }

    int getEntrySafe(const Tensor<int>& t, const MultiIndex& i){
        return t.getEntrySafe(i);
    }

    void functionalUnitTests() {
        
        // Define tensors for tests
        Tensor<int> zeros_3x4x5 = zeros<int>({3, 4, 5});
        Tensor<int> ones_5 = ones<int>({5});
        Tensor<int> twos_5 = constant<int>({5}, 2);
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
        
        std::vector<int> data(24);
        std::iota(data.begin(), data.end(), 1);
        Tensor<int> iota_2x3x4 = iota<int>({2, 3, 4});

        std::vector<MultiIndex> indexes = indexesRowMajor({2, 3, 4});

        BinaryOpTestSuite<Tensor<int>, MultiIndex, int, getEntrySafe> unsafe_indexing_tests;

        int i = 0;
        for(MultiIndex& index : indexes){
            unsafe_indexing_tests.push_back({iota_2x3x4, index, data[i]});
            ++i;
        }
        
        UnaryOpTestSuite<Tensor<int>, Tensor<int>, negInt> neg_tests = {
            {zeros_3x4x5, zeros_3x4x5},
            {ones_3x2x4, t1},
            {ones_10, t2},
            {ones_3x3, t3},
        };

        BinaryOpTestSuite<Tensor<int>, Tensor<int>, Tensor<int>, addInt> add_tests = {
            {ones_5, ones_5, twos_5}
        };

        BinaryOpTestSuite<Tensor<int>, Tensor<int>, Tensor<int>, matmulInt> matmul_tests = {
            {ones_10, ones_10, scalar_10}, // 1D (dot product)
            {ones_5, iota_5, scalar_15}, // 1D (dot product)
            {iota_5, iota_5, scalar_55}, // 1D (dot product)
            {ones_3x3, ones_3x3, threes_3x3}, // 2D (matrix multiplication)
            {iota_3x3, iota_3x3, iota_3x3_squared},
            {iota_2x2x3, iota_3x3, iota_3D_times_2D_result}
            // TO DO: test higher dimensions
        };

        std::cout << "Unsafe Indexing Tests" << std::endl;
        runBinaryOpTestSuite<Tensor<int>, MultiIndex, int, getEntrySafe>(unsafe_indexing_tests);
        std::cout << "Neg Tests" << std::endl;
        runUnaryOpTestSuite<Tensor<int>, Tensor<int>, negInt>(neg_tests);
        std::cout << "Addition Tests" << std::endl;
        runBinaryOpTestSuite<Tensor<int>, Tensor<int>, Tensor<int>, addInt>(add_tests);
        std::cout << "Matmul Tests" << std::endl;
        runBinaryOpTestSuite<Tensor<int>, Tensor<int>, Tensor<int>, matmulInt>(matmul_tests);

    }

}