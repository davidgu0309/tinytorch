#include "../include/layer.hpp"

namespace tinytorch {
    template <typename T>
    Linear<T>::Linear(size_t dim_in, size_t dim_out) {
        T bound = 1./(std::sqrt(dim_in));
        // so that out = x * W and dimensions match
        // x has shape (batch_size, dim_in), so we make W (dim_in, dim_out)
        weights = real_uniform({dim_in, dim_out}, -bound, bound); 
        bias = zeros({dim_out});
    }

    template <typename T>
    Tensor<T> Linear<T>::forward(Tensor<T>& input) { 
        // input has (batch size, dimension) and we assume non-batched input is (dimension), (1, dimension) will not work
        return add(matmul(input, weights_), bias_);  //will require broadcasting for batched input
    }

    template <typename T>
    size_t Linear<T>::num_params() {return dim_in * dim_out;}

    template <typename T>
    Tensor<T>& Linear<T>::weights() {return weights_;}

    template <typename T>
    Tensor<T>& Linear<T>::bias() {return bias_;}

    template <typename T>
    Tensor<T> ReLU<T>::forward(Tensor<T>& input) {
        return relu<T>(input);
    }

}