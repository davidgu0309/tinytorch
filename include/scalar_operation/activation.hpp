/**
 * @file activation.hpp
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

namespace tinytorch {

    /**
     * @struct ScalarReLU
     * 
     * @brief Scalar rectified linear operation with derivative.
     * 
     * Functor-style scalar rectified linear operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarReLU{

        ScalarReLU();

        /**
         * 
         * Scalar rectified linear unit.
         * 
         * @param operands Operand.
         * 
         * @return Rectified linear unit of the operand.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar rectified linear unit with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the rectified linear unit with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

    /**
     * @struct ScalarSigmoid
     * 
     * @brief Scalar sigmoid operation with derivative.
     * 
     * Functor-style scalar sigmoid operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarSigmoid{

        ScalarSigmoid();

        /**
         * 
         * Scalar sigmoid.
         * 
         * @param operands Operand.
         * 
         * @return Sigmoid of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the sigmoid with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the sigmoid with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

} // namespace tinytorch