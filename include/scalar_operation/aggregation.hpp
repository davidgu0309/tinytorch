/**
 * @file aggregation.hpp
 * 
 * @brief Implementation of common scalar operations with derivative.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "concept.hpp"

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{

    /**
     * @struct ScalarAddition
     * 
     * @brief Scalar addition operation with derivative.
     * 
     * Functor-style scalar addition operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarAddition{

        ScalarAddition();

        /**
         * 
         * Scalar addition.
         * 
         * @param operands Operands.
         * 
         * @return Sum of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar addition with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the derivative is computed.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the addition with respect to input input_idx.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarMultiplication
     * 
     * @brief Scalar multiplication operation with derivative.
     * 
     * Functor-style scalar addition operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarMultiplication{

        ScalarMultiplication();

        /**
         * 
         * Scalar multiplication.
         * 
         * @param operands Operands.
         * 
         * @return Product of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar multiplication with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the derivative is computed.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the multiplication with respect to input input_idx.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarArithmeticMean
     * 
     * @brief Scalar arithmetic mean operation with derivative.
     * 
     * Functor-style scalar arithmetic mean operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarArithmeticMean{

        ScalarArithmeticMean();

        /**
         * 
         * Scalar arithmetic mean.
         * 
         * @param operands Operands.
         * 
         * @return Arithmetic mean of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the arithmetic mean with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the arithmetic mean with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    // TODO: other means (e.g. geometri, harmonic)

} // namespace tinytorch

#include "../../src/scalar_operation/aggregation.tpp"