/**
 * @file hadamard.hpp
 * 
 * @brief Componentwise multiplication.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once   

#include "componentwise_operation.hpp"
#include "../scalar_operation/scalar_operation.hpp"

namespace tinytorch{
    
    /**
     * @struct Hadamard
     * 
     * @brief Template for Hadamard operation (componentwise multiplication).
     * 
     * Template for Hadamard operation (componentwise multiplication) with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Hadamard : ComponentwiseOperation<T, ScalarMultiplication<T>> {

        Hadamard();

        /**
         * 
         * Componentwise multiplication of tensors of identical shape. 
         * 
         * @param operands Non-empty vector of operands.
         * 
         * @return Componentwise product of all operand tensors.
         * 
         * @pre The operand vector must contain at least one operand. All operand tensors must have identical shape.
         * 
         **/
        using ComponentwiseOperation<T, ScalarMultiplication<T>>::operator ();

        /**
         * 
         * Gradient of componentwise tensor multiplication with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed.
         * @param operands Operands.
         * 
         * @return Gradient of componentwise multiplication with respect to local input input_idx.
         * 
         **/
        using ComponentwiseOperation<T, ScalarMultiplication<T>>::backward;

    };

} // namespace tinytorch

#include "../../src/tensor_operation/hadamard.tpp"