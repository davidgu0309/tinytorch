#pragma once

namespace tinytorch{

    // Unary operations
    template <typename T>
    T neg(const T& a);

    template <typename T>
    T inv(const T& a);

    // Binary operations
    template <typename T>
    T sum(const T& a, const T& b);

    template <typename T>
    T product(const T& a, const T& b);

}

#include "../src/scalar_operation.tpp"