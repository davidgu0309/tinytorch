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

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{

    // TODO: Doxygen document this concept for scalar ops
    template <typename scalarOp, typename T>
    concept ScalarOperation = requires(scalarOp op, const std::vector<T> operands, size_t idx) {
        { op(operands) } -> std::same_as<T>;                    /** () operator: vector<T> -> T */
        { op.backward(idx, operands) } -> std::same_as<T>;        /** backward(int, vector<T>) -> T */
    };

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

}  // namespace tinytorch

#include "../src/scalar_operation.tpp"