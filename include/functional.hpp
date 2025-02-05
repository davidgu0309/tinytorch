/**
 * @file functional.hpp
 * 
 * @brief Implementation of tensor operations (maybe we can rename this file).
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../include/scalar_operation.hpp"
#include "../include/tensor.hpp"
#include "../include/util.hpp"

#include <cassert>

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch {

    /**
     *
     * @tparam T Operand type.
     * @tparam U Result type.
     * 
     **/
    template<typename T, typename U, U (*unaryOp)(T)>
    Tensor<U> applyUnaryOp(const Tensor<T>& a);

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> neg(const Tensor<T>& a);

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> inv(const Tensor<T>& a);

    /**
     *
     * @tparam T First operand type.
     * @tparam U Second operand type.
     * @tparam V Result type.
     * 
     **/
    template<typename T, typename U, typename V, V (*binaryOp)(T, U)>
    Tensor<V> applyBinaryOp(const Tensor<T>& a, const Tensor<U>& b);

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b);

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b); // {a_1,..,a_k} x {a_k, b_1, b_m} -> {a_1, .., a_k-1, b_1, .., b_m}

    /*
    _ _ _     _ _ _
    _ _ _  x  _ _ _
    _ _ _     _ _ _
    */

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> relu(const Tensor<T>& a);

    /**
     *
     * @tparam T Operand type.
     * 
     **/
    template <typename T>
    Tensor<T> sigmoid(const Tensor<T>& a);

    template <typename T>
    Tensor<T> softmax(const Tensor<T>& a);

    template <typename T>
    Tensor<T> cross_entropy(const Tensor<T>& logits, const Tensor<T>& target);
}

#include "../src/functional.tpp"