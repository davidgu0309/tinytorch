/**
 * @file addition.hpp
 * 
 * @brief Templates for common tensor operations.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once   

#include "componentwise_operation.hpp"
#include "scalar_operation.hpp"

namespace tinytorch{
    
    /**
     * @struct Addition
     * 
     * @brief Template for tensor addition operation.
     * 
     * Template for tensor addition operation with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Addition : ComponentwiseOperation<T, ScalarAddition<T>> {

        Addition();

        /**
         * 
         * Entrywise addition of tensors of identical shape. 
         * 
         * @param operands Non-empty vector of addends.
         * 
         * @return Entrywise sum of all operand tensors.
         * 
         * @pre The operand vector must contain at least one operand. All operand tensors must have identical shape.
         * 
         **/
        using ComponentwiseOperation<T, ScalarAddition<T>>::operator ();

        /**
         * 
         * Gradient of tensor addition with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed.
         * @param operands Operands.
         * 
         * @return Gradient of tensor addition with respect to input input_idx.
         * 
         **/
        using ComponentwiseOperation<T, ScalarAddition<T>>::backward;

    };

} // namespace tinytorch

#include "../src/addition.tpp"