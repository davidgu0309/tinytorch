#pragma once

#include <stddef.h>
#include <vector>

namespace tinytorch {

    size_t numEntries(const std::vector<size_t> shape);

    template<typename T>
    bool isEqual(const T x, const T y, const T eps);
    
}

#include "../src/utils.tpp"


