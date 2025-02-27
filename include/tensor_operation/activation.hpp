/**
 * @file activation.hpp
 * 
 * @brief Componentwise activation functions.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once   

#include "componentwise_operation.hpp"
#include "../scalar_operation/activation.hpp"

namespace tinytorch{
    
    /**
     * @struct ReLU
     * 
     * @brief Template for componentwise ReLU operation.
     * 
     * Template for componentwise ReLU operation with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ReLU : ComponentwiseOperation<T, ScalarReLU<T>> {

        ReLU();

        /**
         * 
         * Componentwise ReLU of tensor. 
         * 
         * @param operands Operand tensor.
         * 
         * @return Componentwise ReLU of operand tensor.
         * 
         * @pre operands.size() == 1
         * 
         **/
        using ComponentwiseOperation<T, ScalarReLU<T>>::operator ();

        /**
         * 
         * Gradient of componentwise ReLU with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed. Useless (TODO: handle this another way)
         * @param operands Operands.
         * 
         * @return Gradient of componentwise ReLU.
         * 
         * @pre operands.size() == 1 && input_idx == 0
         * 
         **/
        using ComponentwiseOperation<T, ScalarReLU<T>>::backward;

    };

    /**
     * @struct Sigmoid
     * 
     * @brief Template for componentwise sigmoid operation.
     * 
     * Template for componentwise sigmoid operation with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Sigmoid : ComponentwiseOperation<T, ScalarSigmoid<T>> {

        Sigmoid();

        /**
         * 
         * Componentwise sigmoid of tensor. 
         * 
         * @param operands Operand tensor.
         * 
         * @return Componentwise sigmoid of operand tensor.
         * 
         * @pre operands.size() == 1
         * 
         **/
        using ComponentwiseOperation<T, ScalarSigmoid<T>>::operator ();

        /**
         * 
         * Gradient of componentwise sigmoid with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed. Useless (TODO: handle this another way)
         * @param operands Operand.
         * 
         * @return Gradient of componentwise sigmoid.
         * 
         * @pre operands.size() == 1 && input_idx == 0
         * 
         **/
        using ComponentwiseOperation<T, ScalarSigmoid<T>>::backward;

    };

} // namespace tinytorch

#include "../../src/tensor_operation/activation.tpp"