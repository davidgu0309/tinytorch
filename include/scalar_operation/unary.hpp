/**
 * @file unary.hpp
 * 
 * Unary meaning taking one argument.
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
     * @struct ScalarSgn
     * 
     * @brief Scalar sign operation with derivative.
     * 
     * Functor-style scalar sign operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSgn{

        ScalarSgn();

        /**
         * 
         * Scalar sign.
         * 
         * @param operands Operand.
         * 
         * @return Sign of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the sign with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the sign with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarAbs
     * 
     * @brief Scalar absolute value operation with derivative.
     * 
     * Functor-style scalar absolute value operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarAbs{

        ScalarAbs();

        /**
         * 
         * Scalar absolute value.
         * 
         * @param operands Operand.
         * 
         * @return Absolute value of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the absolute value with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the absolute value with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarNeg
     * 
     * @brief Scalar negation operation with derivative.
     * 
     * Functor-style scalar negation operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarNeg{

        ScalarNeg();

        /**
         * 
         * Scalar negation.
         * 
         * @param operands Operand.
         * 
         * @return Negation of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the negation with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the negation with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };



}  // namespace tinytorch

#include "../../src/scalar_operation/scalar_operation.tpp"