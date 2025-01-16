#include "../include/utils.h"

namespace tinytorch {
    template <typename T>
    bool isEqualShape(const Tensor<T>& a, const Tensor<T>& b) {
        return a.shape() == b.shape();
    }

    size_t numEntries(std::vector<size_t> shape) {
        if (shape.size() == 0) return 0;
        size_t result = 1;
        for (int i=0; i<shape.size(); i++) {
            result = result * shape[i];
        }
        return result;
    }

}