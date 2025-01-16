#pragma once

#include <vector>
#include <memory>
#include "../include/utils.h"

namespace tinytorch {

template <typename T>
class Tensor {
    public:
        Tensor();
        Tensor(const std::vector<size_t> shape);
        Tensor(const std::vector<T>& data,
                const std::vector<size_t> shape);        

        std::vector<size_t> shape() const;
        size_t size() const;

        std::vector<T>* data();
        const std::vector<T>* data();

        std::vector<T>* grad();
        const std::vector<T>* grad();

        bool operator==(const Tensor<T>& other);

        //add(Tensor other);
        // {
        //     if (requires_grad) {

        //     }
        //     grad 
        // }

    private:
        std::unique_ptr<std::vector<T>> data_;
        std::vector<size_t> shape_;
        std::unique_ptr<std::vector<T>> grad_;
        bool requires_grad;
};

template <typename T>
Tensor<T> zeros(const std::vector<size_t> shape);

template <typename T>
Tensor<T> ones(const std::vector<size_t> shape);

}

#include "../src/tensor.cpp"

