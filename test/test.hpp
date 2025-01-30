/**
 * @file test.hpp
 * 
 * @brief Testing framework.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */

#pragma once

#include "function_traits.hpp"

/**
 * @namespace test
 * 
 * @brief Namespace of the testing framework.
 * 
 */
namespace test{

    template<typename T>
    void consoleLog(const T& result, const T& correct);

    // struct Test{}; // Maybe think about this

    
    /**
     * @struct UnitTest
     * 
     * Structure for unit tests.
     * 
     */
    template<typename FunctionToTest>
    struct UnitTest {
        /**
         * Get types.
         */
        using FT = FunctionTraits<FunctionToTest>;
        using ReturnType = typename FT::ReturnType;
        using ArgumentTypes = typename FT::ArgumentTypes;

        ArgumentTypes operands; /** Test operands. */
        ReturnType correct_result; /** Correct test result. */

        /**
         * 
         * Runs the unit test, i.e. checks if FunctionToTest(operands) == result.
         * 
         * @todo How should we handle logging? Could use flag (e.g. via Makefile) or argument
         * 
         * @return FunctionToTest(operands) == result
         * 
         */
        bool run() const;
    };

    /**
     * @brief Test suite for FunctionToTest.
     * 
     * @tparam FunctionToTest Function to test.
     * 
     */
    template<typename FunctionToTest>
    struct TestSuite {
        std::vector<UnitTest<FunctionToTest>> unit_tests;
        void addTest(UnitTest<FunctionToTest> unit_test);
        void run() const;
    };

} // namespace test

#include "test.tpp"