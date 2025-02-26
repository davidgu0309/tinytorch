/**
 * @file loss.hpp
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
     * 
     * @todo Better comments and names in LaTeX.
     * 
     * @struct ScalarLp
     * 
     * @brief Scalar l^p operation with derivative.
     * 
     * Functor-style scalar l^p operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarLp{

        ScalarLp();

        /**
         * 
         * Scalar l^p.
         * 
         * @param operands Operand.
         * 
         * @return l^p operation of the operands.
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the l^p with respect to input input_idx 0.
         * 
         * @param input_idx Unnecessary (TODO: handle unary and binary ops separately).
         * 
         * @param operands Operand (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the l^p with respect to input input_idx.
         * 
         * @pre operands.size() == 2
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };
    
} // namespace tinytorch