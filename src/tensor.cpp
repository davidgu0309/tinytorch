#include "../include/tensor.h"

namespace tinytorch {
    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> shape) : shape_(shape) {}

    template <typename T>
    Tensor<T>::Tensor(const std::vector<T>& data, 
        const std::vector<size_t> shape) : shape_(shape)
    {
        //std::vector<T>* new_data = malloc(numElements * sizeof(T));
        std::vector<T>* new_data = new std::vector(data);
        data_ = std::unique_ptr<std::vector<T>>(new_data);
    }

    template <typename T>
    std::vector<size_t> Tensor<T>::shape() const {return shape_;}
        
    template <typename T>
    std::vector<T>* Tensor<T>::data() {return data_.get();}

    template <typename T>
    const std::vector<T>* Tensor<T>::data() const {return data_.get();}

    template <typename T>
    std::vector<T>* Tensor<T>::grad() {return grad_.get();}

    template <typename T>
    bool Tensor<T>::operator== (const Tensor<T>& other) {
        std::vector<T> other_data = *other.data();
        return (*data() == other_data) && isEqualShape(*this, other);    //can we deference unique_ptr?
    }


    template <typename T>
    const std::vector<T>* Tensor<T>::grad() const {return grad_.get();}

    template <typename T>
    size_t Tensor<T>::size() const {
        if (shape_.size() == 0) return 0;
        size_t result = 1;
        for (int i=0; i<shape_.size(); i++) {
            result = result * shape_[i];
        }
        return result;
    }


    template <typename T>
    Tensor<T> zeros(const std::vector<size_t> shape) {
        std::vector<T> zero_vector(numEntries(shape), 0);
        return Tensor(zero_vector, shape); 
    }

    template <typename T>
    Tensor<T> ones(const std::vector<size_t> shape) {
        std::vector<T> zero_vector(numEntries(shape), 1);
        return Tensor(zero_vector, shape); 
    }
}
