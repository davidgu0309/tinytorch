#pragma once
#include "../include/tensor.h"

namespace tinytorch {
    template <typename T>
    bool isEqualShape(const Tensor<T>& a, const Tensor<T>& b);

    size_t numEntries(const std::vector<size_t>& shape);
}


