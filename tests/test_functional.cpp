#include "../include/functional.h"
#include "../include/tensor.h"

namespace tinytorch {
    bool testAddTensors() {
        Tensor<int> ex1_t1 = Tensor<int>(std::vector<int>({1,2,3}), std::vector<size_t>({3}));
        Tensor<int> ex1_t2 = ones<int>(std::vector<size_t>({3}));
        Tensor<int> ex1_sum = Tensor<int>(std::vector<int>({2,3,4}), std::vector<size_t>({3}));
        return ex1_sum == add(ex1_t1, ex1_t2);
    }

}