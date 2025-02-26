/**
 * @file trigonometry.hpp
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

// #include <cmath>

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{

    /**
     * @struct ScalarSin
     * 
     * @brief Scalar sine operation with derivative.
     * 
     * Functor-style scalar sine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSin{

        ScalarSin();

        /**
         * 
         * Scalar sine.
         * 
         * @param operands Operand.
         * 
         * @return Sine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the sine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the sine with respect to input input_idx.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarCos
     * 
     * @brief Scalar cosine operation with derivative.
     * 
     * Functor-style scalar cosine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarCos{

        ScalarCos();

        /**
         * 
         * Scalar cosine.
         * 
         * @param operands Operand.
         * 
         * @return Cosine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the cosine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the cosine with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarTan
     * 
     * @brief Scalar tangent operation with derivative.
     * 
     * Functor-style scalar tangent operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarTan{

        ScalarTan();

        /**
         * 
         * Scalar tangent.
         * 
         * @param operands Operand.
         * 
         * @return Tangent of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the tanget with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the tangent with respect to input input_idx.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    // TODO: inverse trig

} // namespace tinytorch

#include "../../src/scalar_operation/trigonometry.tpp"