/**
 * @file matmul.hpp
 * 
 * @brief Templates for common tensor operations.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "tensor_operation.hpp"

namespace tinytorch{

template <typename T>
struct Matmul : TensorOperation<T> {

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
    Tensor<T> operator()(const std::vector<Tensor<T>> operands) const override;

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
    Tensor<T> backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const override;
    
};

} // namepsace tinytorch

#include "../../src/tensor_operation/matmul.tpp"