/**
 * @file scalar_operation.hpp
 * 
 * @brief Templated scalar operations.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include <cmath>

/**
 * @namespace tensor
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tensor{

    // Unary operations
    template <typename T>
    T neg(const T x);

    template <typename T>
    T inv(const T x);

    template <typename T>
    T relu(const T x);

    template <typename T>
    T sigmoid(const T x);

    // Binary operations
    template <typename T>
    T sum(const T x, const T y);

    template <typename T>
    T product(const T x, const T y);

}

#include "../src/scalar_operation.tpp"