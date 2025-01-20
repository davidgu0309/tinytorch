#pragma once

#include "../include/tensor.hpp"
#include "../include/functional.hpp"

#include <vector>
#include <memory>
#include <cmath>

namespace tinytorch{

template <typename T>
class Layer {
    virtual Tensor<T> forward(Tensor<T>& input);
};

// class model { def forward(): blablabla return output}
// logits = model(input)
// loss = function(logits, targets)
// loss.backward()

// if (input.ndim() == 3) batches   (B, S, T)
// if (input.ndim() == 2) no batch  (S, T)

template <typename T>
class Linear: Layer<T> {
    public:
        Linear(size_t dim_in, size_t dim_out);      //Kaiming uniform weights initialization
        
        Tensor<T> forward(Tensor<T>& input);

        size_t num_params();
        Tensor<T>& weights();
        Tensor<T>& bias();

    private:
        size_t dim_in, dim_out;
        Tensor<T> weights_;    //maybe std::unique_ptr(Tensor<T>), shape dim_out x dim_in
        Tensor<T> bias_;        //shape dim_out

};

template <typename T>
class ReLU {
    public:
        Tensor<T> forward(Tensor<T>& input);
};

template <typename T>
class Conv1D {
    public:
        Tensor<T> forward(Tensor<T>& input);

    // TODO -----------complete -----------
};

}

#include "../src/layer.tpp"
