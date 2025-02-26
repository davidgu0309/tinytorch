/**
 * @file power.hpp
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

#include <cmath>

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{

    /**
     * @struct ScalarPow
     * 
     * @brief Scalar power operation with derivative.
     * 
     * Functor-style scalar power operation differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ScalarPow{

        ScalarPow();

        /**
         * 
         * Scalar power.
         * 
         * @param operands Operands.
         * 
         * @return Scalar power operands[0] ^ operands[1].
         * 
         **/
        T operator()(const std::vector<T> operands) const;

        /**
         * 
         * Partial derivative of the scalar power operands[0] ^ operands[1].
         * 
         * @param input_idx 0 for base, 1 for exponent.
         * 
         * @param operands Operands (point at which the derivative is evaluated).
         * 
         * @return Partial derivative of the power with respect to input input_idx.
         * 
         * @pre operands.size() == 2
         * 
         **/
        T backward(const size_t input_idx, const std::vector<T> operands) const;

    };

} // namespace tinytorch

#include "../../src/scalar_operation/power.tpp"