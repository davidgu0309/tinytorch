#pragma once

#include "../include/utils.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>
#include <functional>
#include "distribution.hpp"

namespace tinytorch {

    typedef std::vector<size_t> Shape;
    typedef std::vector<size_t> MultiIndex; // 0-based multiindexes

    bool multiIndexLegalityTest(const Shape shape, const MultiIndex multi_index);

    std::ostream& operator << (std::ostream& out, const MultiIndex& index);

    template <typename T>
    class Tensor {

        // Default visibility is private
        Shape shape_;
        std::unique_ptr<std::vector<T>> data_;
        std::unique_ptr<std::vector<T>> grad_;
        bool requires_grad;

    public:

        Tensor(const T value);    // Returns scalar (shape {})
        Tensor(const std::vector<size_t> shape);
        Tensor(const std::vector<size_t> shape,
                    const std::vector<T>& data);        

        size_t size() const;
        Shape shape() const;

        std::vector<T>* data();
        std::vector<T>* grad();

        const std::vector<T>* data() const;
        const std::vector<T>* grad() const;

        T& getEntryUnsafe(MultiIndex index);
        const T& getEntryUnsafe(MultiIndex index) const;

        T& getEntrySafe(MultiIndex index);
        const T& getEntrySafe(MultiIndex index) const;

        // Comparison operators
        bool operator == (const Tensor<T>& other) const;
        bool shapeEqual(const Tensor<T>& other) const;

        template <typename U>
        friend std::ostream& operator << (std::ostream& out, const Tensor<U>& tensor);

    };

    // Common tensors
    template <typename T>
    Tensor<T> zeros(const std::vector<size_t> shape);

    template <typename T>
    Tensor<T> ones(const std::vector<size_t> shape);

    template <typename T>
    Tensor<T> constant(const std::vector<size_t> shape, T value);

    template <typename T>
    Tensor<T> iota(const std::vector<size_t> shape);

    // TO DO: identity

    // TO DO: random
    template <typename T>
    Tensor<T> initialize_using_generator(const std::vector<size_t> shape, std::function<T()> generator);
}

#include "../src/tensor.tpp"

