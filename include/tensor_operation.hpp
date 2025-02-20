/**
 * @file tensor_operation.hpp
 * 
 * @brief Templates for common tensor operations.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../tensor/include/functional.hpp"
#include "../tensor/include/tensor.hpp"

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
     * Abstract functor-style structure for tensor operations differentiable with respect to parameters
     * and with respect to inputs.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct TensorOperation {
        std::vector<Tensor<T>> parameters_; // TO DO: factor out parameters
        // bool requires_grad_;

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
        virtual Tensor<T> operator()(std::vector<Tensor<T>>& operands) const = 0;

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
        virtual std::vector<Tensor<T>> backwardWRTInputs(size_t input_idx, std::vector<Tensor<T>>& operands) const = 0;

        /**
         * 
         * Gradient of tensor operation with respect to parameter parameter_idx.
         * 
         * @param input_idx Index of parameter with respect to which the gradient is computed.
         * @param operands Operands.
         * 
         * @return Gradient of tensor operation with respect to parameter parameter_idx.
         * 
         **/
        virtual std::vector<Tensor<T>> backwardWRTParameters(size_t parameter_idx, std::vector<Tensor<T>>& operands) const = 0;

    };

    /**
     * @struct TensorAddition
     * 
     * @brief Template for tensor addition operation.
     * 
     * Template for tensor addition operation with gradient computation.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct TensorAddition : TensorOperation<T> {

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
        Tensor<T> operator()(std::vector<Tensor<T>>& operands) const;

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
        Tensor<T> backwardWRTInputs(size_t input_idx, std::vector<Tensor<T>>& operands) const;

        /**
         * 
         * Useless, might want to implement struct for tensor operations without params.
         * 
         **/
        Tensor<T> backwardWRTParameters(size_t parameter_idx, std::vector<Tensor<T>>& operands) const;
    };

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
        Tensor<T> operator()(std::vector<Tensor<T>>& operands) const;

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
        Tensor<T> backwardWRTInputs(size_t input_idx, std::vector<Tensor<T>>& operands) const;

        /**
         * 
         * Useless, might want to implement struct for tensor operations without params.
         * 
         **/
        Tensor<T> backwardWRTParameters(size_t parameter_idx, std::vector<Tensor<T>>& operands) const;
    };

}  // namespace tinytorch

#include "../src/tensor_operation.tpp"