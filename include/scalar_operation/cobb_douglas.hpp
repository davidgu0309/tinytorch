/**
 * @file cobb_douglas.hpp
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
     * @struct ScalarCobbDouglas
     * 
     * @brief Scalar Cobb-Douglas operation with derivative.
     * 
     * Functor-style scalar Cobb-Douglas operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarCobbDouglas{

        ScalarCobbDouglas();

        /**
         * 
         * Scalar Cobb-Douglas.
         * 
         * @param operands Operands.
         * 
         * @return Cobb-Douglas of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the Cobb-Douglas with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the Cobb-Douglas with respect to input input_idx.
         * 
         * @pre operands.size() == 1
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };
    
} // namespace tinytorch

#include "../../src/scalar_operation/cobb_douglas.tpp"