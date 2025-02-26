/**
 * @file concept.hpp
 * 
 * @brief Concept for scalar operations with derivative.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch{
    // TODO: Doxygen document this concept for scalar ops
    // TODO: Is it possible to create specializations for unary and binary ops to optimize?
    template <typename scalarOp, typename T>
    concept ScalarOperation = requires(scalarOp op, const std::vector<T> operands, size_t idx) {
        { op(operands) } -> std::same_as<T>;                    /** () operator: vector<T> -> T */
        { op.backward(idx, operands) } -> std::same_as<T>;        /** backward(int, vector<T>) -> T */
    };
}