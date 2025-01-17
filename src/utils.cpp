#include "../include/utils.hpp"

namespace tinytorch {

    size_t numEntries(const std::vector<size_t> shape) {
        if (shape.size() == 0) return 0;
        size_t result = 1;
        for (int i=0; i<shape.size(); i++) {
            result = result * shape[i];
        }
        return result;
    }

}