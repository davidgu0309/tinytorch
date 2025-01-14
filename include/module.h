#pragma once
#include <layer.h>

namespace tinytorch {

template <typename T>
class Sequential {
    public:
        Sequential(std::vector<Layer<T>> layerList);

        forward(const Tensor<T>& input); 
        // {
        //     for (Layer<T> L: layerList) {
        //         input = L.forward(input);
        //     }
        // }
};

}

