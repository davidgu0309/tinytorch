#include "../include/util.hpp"

namespace tinytorch {

    // shape {} models scalars
    size_t numEntries(const std::vector<size_t> shape) {
        size_t result = 1;
        for (int i=0; i<shape.size(); i++) {
            result = result * shape[i];
        }
        return result;
    }

}