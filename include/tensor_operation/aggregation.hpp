/**
 * @file hadamard.hpp
 * 
 * @brief Componentwise multiplication.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once   

#include "aggregation_operation.hpp"
#include "../scalar_operation/aggregation.hpp"

namespace tinytorch{
    
    /**
     * @struct Sum
     * 
     * @brief Template for sum operation (sum of all entries).
     * 
     * Template for sum operation (sum of all entries) with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Sum : AggregationOperation<T, ScalarAddition<T>> {

        Sum();

        /**
         * 
         * Sum of entries.
         * 
         * @param operand Operand tensor.
         * 
         * @return Sum of all entries.
         * 
         * @pre The operand vector must contain exactly one tensor.
         * 
         **/
        using AggregationOperation<T, ScalarAddition<T>>::operator ();

        /**
         * 
         * Gradient.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed (must be 0)
         * @param operand Operands.
         * 
         * @return Gradient of sum of entries.
         * 
         **/
        using AggregationOperation<T, ScalarAddition<T>>::backward;

    };

    /**
     * @struct Product
     * 
     * @brief Template for product operation (product of all entries).
     * 
     * Template for product operation (product of all entries) with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Product : AggregationOperation<T, ScalarMultiplication<T>> {

        Product();

        /**
         * 
         * Product of entries.
         * 
         * @param operand Operand tensor.
         * 
         * @return Product of all entries.
         * 
         * @pre The operand vector must contain exactly one tensor.
         * 
         **/
        using AggregationOperation<T, ScalarMultiplication<T>>::operator ();

        /**
         * 
         * Gradient.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed (must be 0)
         * @param operand Operands.
         * 
         * @return Gradient of product of entries.
         * 
         **/
        using AggregationOperation<T, ScalarMultiplication<T>>::backward;

    };

    /**
     * @struct ArithmeticMean
     * 
     * @brief Template for arithmetic mean operation (arithmetic mean of all entries).
     * 
     * Template for arithmetic mean operation (arithmetic mean of all entries) with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ArithmeticMean : AggregationOperation<T, ScalarArithmeticMean<T>> {

        ArithmeticMean();

        /**
         * 
         * Arithmetic mean of entries.
         * 
         * @param operand Operand tensor.
         * 
         * @return Arithmetic mean of all entries.
         * 
         * @pre The operand vector must contain exactly one tensor.
         * 
         **/
        using AggregationOperation<T, ScalarArithmeticMean<T>>::operator ();

        /**
         * 
         * Gradient.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed (must be 0)
         * @param operand Operands.
         * 
         * @return Gradient of arithmetic mean of entries.
         * 
         **/
        using AggregationOperation<T, ScalarArithmeticMean<T>>::backward;

    };

} // namespace tinytorch

#include "../../src/tensor_operation/aggregation.tpp"