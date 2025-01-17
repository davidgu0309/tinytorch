#pragma once

#include "../include/utils.hpp"

#include <iostream>
#include <vector>
#include <memory>

namespace tinytorch {

 /*  TO DELETE
    typedef std::vector<size_t> Multiindex 

    std::vector<Multiindex> stack;
    stack
    for(int d : shape){
        for(int i = 0; i < d; ++i){
            for(auto mi : stack){
                stack.push_back()
            }
        }
    }

    shape von matmua(a, b) a.shap[0:-1] + b.shape[1:]

    shape von transpose(a) ist a.shape.reverse
    */

    template <typename T>
    class Tensor {

        // Default visibility is private
        std::vector<size_t> shape_;
        std::unique_ptr<std::vector<T>> data_;
        std::unique_ptr<std::vector<T>> grad_;
        bool requires_grad;

    public:

        Tensor();
        Tensor(const std::vector<size_t> shape);
        Tensor(const std::vector<T>& data,
                const std::vector<size_t> shape);        

        size_t size() const;
        std::vector<size_t> shape() const;

        std::vector<T>* data();
        std::vector<T>* grad();

        const std::vector<T>* data() const;
        const std::vector<T>* grad() const;

        // Comparison operators
        bool operator == (const Tensor<T>& other) const;
        bool shapeEqual(const Tensor<T>& other) const;

        template <typename U>
        friend std::ostream& operator << (std::ostream& out, const Tensor<U>& tensor);

    };

    // Common tensors
    template <typename T>
    Tensor<T>& zeros(const std::vector<size_t> shape);

    template <typename T>
    Tensor<T>& ones(const std::vector<size_t> shape);

}

#include "../src/tensor.tpp"

