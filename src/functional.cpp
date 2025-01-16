#include "../include/functional.h"

namespace tinytorch {
    template <typename T>
    Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
        assert(isEqualShape(a, b));
        Tensor<T> result(a.shape());    //check if need malloc
        std::vector<T> result_data = *result.data();
        std::vector<T> dataA = *a.data();   //check if use const
        std::vector<T> dataB = *b.data();   
        
        for (int i=0; i<dataA.size(); i++) {
            result_data[i] = dataA[i] + dataB[i];
        }
        return result;
    }


}
