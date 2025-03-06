/**
 * @file power.hpp
 * 
 * @brief Componentwise power operation.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once   

#include "componentwise_operation.hpp"
#include "../scalar_operation/power.hpp"

namespace tinytorch{
    
    /**
     * @struct Power
     * 
     * @brief Template for componentwise power operation.
     * 
     * Template for componentwise power with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Power : ComponentwiseOperation<T, ScalarPow<T>> {

        Power();

        /**
         * 
         * Componentwise power of tensor. 
         * 
         * @param operands operand[0] bases, operand[1] exponents.
         * 
         * @return Componentwise power.
         * 
         * @pre operands.size() == 2 && operands[0].shape() == operands[1].shape()
         * 
         **/
        using ComponentwiseOperation<T, ScalarPow<T>>::operator ();

        /**
         * 
         * Gradient of componentwise power with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed.
         * @param operands Operands.
         * 
         * @return Gradient of componentwise power with respect to input input_idx.
         * 
         **/
        using ComponentwiseOperation<T, ScalarPow<T>>::backward;

    };

} // namespace tinytorch

#include "../../src/tensor_operation/power.tpp"