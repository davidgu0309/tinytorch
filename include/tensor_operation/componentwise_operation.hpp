/**
 * @file componentwise_operation.hpp
 * 
 * @brief Template for componentwise tensor operations.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../scalar_operation/scalar_operation.hpp"
#include "../../tensor/include/tensor.hpp"

using namespace tensor;

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{

    /**
     * @struct TensorOperation
     * 
     * @brief Interface for componentwise tensor operations.
     * 
     * Functor-style structure for componentwise tensor operations differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T, ScalarOperation<T> ScalarOperation>
    struct ComponentwiseOperation : TensorOperation<T>{

        ScalarOperation scalarOperation_;

        ComponentwiseOperation();

        /**
         * 
         * Componentwise tensor operation.
         * 
         * @param operands Operands.
         * 
         * @return Result of the componentwise tensor operation with the passed operands.
         * 
         **/
        Tensor<T> operator()(const std::vector<Tensor<T>> operands) const override;

        /**
         * 
         * Gradient of the componentwise tensor operation with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed.
         * 
         * @param operands Operands.
         * 
         * @return Gradient of tensor operation with respect to input input_idx.
         * 
         **/
        Tensor<T> backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const override;

    };

}  // namespace tinytorch

#include "../../src/tensor_operation/componentwise_operation.tpp"