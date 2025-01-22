namespace tinytorch {
    template <typename T>
    Linear<T>::Linear(const size_t dim_in, const size_t dim_out) : dim_in_(dim_in), dim_out_(dim_out){
        T bound = 1./(std::sqrt(((T)dim_in)));
        // so that out = x * W and dimensions match
        // x has shape (batch_size, dim_in), so we make W (dim_in, dim_out)
        weights_ = real_uniform<T>({dim_in, dim_out}, -bound, bound); 
        bias_ = zeros<T>({dim_out});
    }

    template <typename T>
    Tensor<T> Linear<T>::forward(const Tensor<T>& input) { 
        // input has (batch size, dimension) and we assume non-batched input is (dimension), (1, dimension) will not work
        // TO DO: will require broadcasting for batched input
        return add<T>(matmul<T>(input, weights_), bias_); 
    }

    template <typename T>
    size_t Linear<T>::num_params() const {
        return dim_in_ * dim_out_ + dim_out_;  // TO DO: decide if we want to count bias params
    }

    template <typename T>
    Tensor<T>& Linear<T>::weights() {
        return weights_;
    }

    template <typename T>
    Tensor<T>& Linear<T>::bias() {
        return bias_;
    }

    template <typename T>
    Tensor<T> ReLU<T>::forward(const Tensor<T>& input) {
        return relu<T>(input);
    }

}