#pragma once

#include <tensor.h>
#include <vector>
#include <memory>
#include <functional.h>

namespace tinytorch{

template <typename T>
class Layer {
    virtual forward(Tensor<T>& input);
};

// class model { def forward(): blablabla return output}
// logits = model(input)
// loss = function(logits, targets)
// loss.backward()

// if (input.ndim() == 3) batches   (B, S, T)
// if (input.ndim() == 2) no batch  (S, T)

template <typename T>
class Linear: Layer {
    public:
        Linear(size_t dim_in, size_t dim_out);
        
        Tensor<T> forward(Tensor<T>& input);

        size_t num_params() const {return dim_in * dim_out;}
        T* weights() {return weights_.get();}
        const T* bias() const {return bias_.get();}

    private:
        size_t dim_in, dim_out;
        std::unique_ptr<T> weights_;
        std::unique_ptr<T> bias_;

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