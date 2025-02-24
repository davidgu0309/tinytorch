/**
 * @file tensor_operation.hpp
 * 
 * @brief Template for tensor operations.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../../tensor/include/functional.hpp"
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
     * @brief Interface for tensor operations.
     * 
     * Functor-style structure for tensor operations differentiable with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct TensorOperation {

        TensorOperation();

        /**
         * 
         * Tensor operation.
         * 
         * @param operands Operands.
         * 
         * @return Result of the tensor operation with the passed operands.
         * 
         **/
        virtual Tensor<T> operator()(const std::vector<Tensor<T>> operands) const = 0;

        /**
         * 
         * Gradient of tensor operation with respect to input input_idx.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed.
         * @param operands Operands.
         * 
         * @return Gradient of tensor operation with respect to input input_idx.
         * 
         **/
        virtual Tensor<T> backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const = 0;

    };

}  // namespace tinytorch

#include "../src/tensor_operation.tpp"