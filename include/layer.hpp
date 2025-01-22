#pragma once

#include "../include/tensor.hpp"
#include "../include/functional.hpp"

#include <vector>
#include <memory>
#include <cmath>

namespace tinytorch{

template <typename T>
class Layer {
    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
};

// class model { def forward(): blablabla return output}
// logits = model(input)
// loss = function(logits, targets)
// loss.backward()

// if (input.ndim() == 3) batches   (B, S, T)
// if (input.ndim() == 2) no batch  (S, T)

template <typename T>
class Linear: Layer<T> {

        size_t dim_in_, dim_out_;
        Tensor<T> weights_;    //maybe std::unique_ptr(Tensor<T>), shape (dim_in, dim_out)
        Tensor<T> bias_;    //shape {dim_out_}

    public:
        // Kaiming uniform weights initialization
        Linear(const size_t dim_in, const size_t dim_out);      
        
        Tensor<T> forward(const Tensor<T>& input);

        size_t num_params() const; // For now does not count bias weights
        Tensor<T>& weights();
        Tensor<T>& bias();
               

};

template <typename T>
class ReLU {
    public:
        Tensor<T> forward(const Tensor<T>& input);
};

template <typename T>
class Conv1D {
    public:
        Tensor<T> forward(const Tensor<T>& input);

    // TODO -----------complete -----------
};

}

#include "../src/layer.tpp"
