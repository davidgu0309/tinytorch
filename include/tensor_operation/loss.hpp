/**
 * @file loss.hpp
 * 
 * @brief Componentwise loss functions.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once   

#include "componentwise_operation.hpp"
#include "../scalar_operation/loss.hpp"

namespace tinytorch{
    
    /**
     * @struct Lp
     * 
     * @brief Template for componentwise Lp operation.
     * 
     * Template for componentwise Lp operation with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct Lp : ComponentwiseOperation<T, ScalarLp<T>> {

        Lp();

        /**
         * 
         * Componentwise Lp of tensor. 
         * 
         * @param operands Operand tensor.
         * 
         * @return Componentwise Lp of operand tensor.
         * 
         * @todo Actually implement Lp, maybe make separate L2.
         * 
         * @pre operands.size() == 1
         * 
         **/
        using ComponentwiseOperation<T, ScalarLp<T>>::operator ();

        /**
         * 
         * Gradient of componentwise Lp with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed. Useless (TODO: handle this another way)
         * @param operands Operands.
         * 
         * @return Gradient of componentwise Lp.
         * 
         **/
        using ComponentwiseOperation<T, ScalarLp<T>>::backward;

    };

} // namespace tinytorch

#include "../../src/tensor_operation/loss.tpp"