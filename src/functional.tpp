#include "../include/utils.hpp"

namespace tinytorch {
    
    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        assert(a.shapeEqual(b));
        Tensor<T> result(a.shape());    // Do we need new? No, as long as we return a copy and not a reference!
        std::vector<T>& result_data = *result.data();
        std::vector<T> dataA = *a.data();
        std::vector<T> dataB = *b.data();   
        for (int i=0; i<dataA.size(); i++) {
            result_data[i] = dataA[i] + dataB[i];
        }
        return result;
    }


}
