/**
 * @file scalar_operation.hpp
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
     * @struct ScalarSinH
     * 
     * @brief Scalar hyperbolic sine operation with derivative.
     * 
     * Functor-style scalar hyperbolic sine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSinH{

        ScalarSinH();

        /**
         * 
         * Scalar hyperbolic sine.
         * 
         * @param operands Operand.
         * 
         * @return Hyperbolic sine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the hyperbolic sine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the hyperbolic sine with respect to input.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarCosH
     * 
     * @brief Scalar hyperbolic cosine operation with derivative.
     * 
     * Functor-style scalar hyperbolic cosine operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarCosH{

        ScalarCosH();

        /**
         * 
         * Scalar hyperbolic cosine.
         * 
         * @param operands Operand.
         * 
         * @return Hyperbolic cosine of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the hyperbolic cosine with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the hyperbolic cosine with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarTanH
     * 
     * @brief Scalar hyperbolic tangent operation with derivative.
     * 
     * Functor-style scalar hyperbolic tangent operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarTanH{

        ScalarTanH();

        /**
         * 
         * Scalar hyperbolic tangent.
         * 
         * @param operands Operand.
         * 
         * @return Hyperbolic tangent of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the hyperbolic tanget with respect to input.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the hyperbolic tangent with respect to input input_idx.
         * 
         * @pre operands.size() == 1.
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    // TODO: inverse hyperbolic trig

} // namespace tinytorch

#include "../../src/scalar_operation/hyperbolic_trigonometry.tpp"