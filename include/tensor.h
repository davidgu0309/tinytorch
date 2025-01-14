#pragma once

#include <vector>
#include <memory>

namespace tinytorch {

template <typename T>
class Tensor {
    public:
        Tensor();
        Tensor(const std::vector<size_t>& shape);
        Tensor(const std::vector<T>& data,
                const std::vector<size_t> shape);

        const std::vector<size_t>& shape() const {return shape_;}
        size_t size() const;
        T* data() {return data_.get();}
        const T* data() const {return data_.get();}

        add(Tensor other);
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
}

