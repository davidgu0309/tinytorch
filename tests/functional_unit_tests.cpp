#include "../include/functional.hpp"
#include "../include/tensor.hpp"

#include <iostream>

namespace tinytorch {

    bool additionTest(const Tensor<int>& t1, const Tensor<int>& t2, const Tensor<int>& result) {
        Tensor<int> sum = add(t1, t2);
        std::cout << sum << std::endl;
        return sum == result;
    }

    bool additionTest1(){
        Tensor<int> t1 = Tensor<int>(std::vector<int>({1,2,3}), std::vector<size_t>({3}));
        std::cout << t1 << std::endl;
        Tensor<int> t2 = ones<int>(std::vector<size_t>({3}));
        std::cout << t2 << std::endl;
        Tensor<int> result = Tensor<int>(std::vector<int>({2,3,4}), std::vector<size_t>({3}));
        std::cout << result << std::endl;
        return additionTest(t1, t2, result);
    }

}