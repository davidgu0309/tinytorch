/**
 * @file aggregation_operation.hpp
 * 
 * @brief Template for aggregation tensor operations (operations that reduce the tensor to a scalar tensor).
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../scalar_operation/concept.hpp"
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
     * Functor-style structure for differentiable aggregation tensor operations.
     * 
     * @tparam T Floating point data type for numerical computations.
     * @tparam ScalarOperation Scalar aggregation operation.
     * 
     **/
    template <typename T, ScalarOperation<T> ScalarOperation>
    struct AggregationOperation : TensorOperation<T>{

        ScalarOperation scalarOperation_;

        AggregationOperation();

        /**
         * 
         * Aggregation operation.
         * 
         * @param operand Operand tensor.
         * 
         * @return Result of the aggregation of the entries of the operand tensor.
         * 
         **/
        Tensor<T> operator()(const std::vector<Tensor<T>> operand) const override;

        /**
         * 
         * Gradient of the aggregation operation.
         * 
         * @param input_idx Index of input with respect to which the gradient is computed (must be 0).
         * 
         * @param operand Operand.
         * 
         * @return Gradient of aggregation operation.
         * 
         **/
        Tensor<T> backward(const size_t input_idx, const std::vector<Tensor<T>> operand) const override;

    };

}  // namespace tinytorch

#include "../../src/tensor_operation/aggregation_operation.tpp"