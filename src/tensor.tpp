namespace tinytorch {

    // Constructors
    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t> shape) : shape_(shape) {
        std::vector<T>* empty_data = new std::vector<T>(numEntries(shape));
        data_ = std::unique_ptr<std::vector<T>>(empty_data);
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<T>& data, 
        const std::vector<size_t> shape) : shape_(shape)
    {
        //std::vector<T>* new_data = malloc(numElements * sizeof(T));
        std::vector<T>* new_data = new std::vector(data);
        data_ = std::unique_ptr<std::vector<T>>(new_data);
    }



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
    std::vector<size_t> Tensor<T>::shape() const {
        return shape_;
    }
        


    template <typename T>
    std::vector<T>* Tensor<T>::data() {
        return data_.get();
    }

    template <typename T>
    std::vector<T>* Tensor<T>::grad() {
        return grad_.get();
    }


    template <typename T>
    const std::vector<T>* Tensor<T>::data() const {
        return data_.get();
    }

    template <typename T>
    const std::vector<T>* Tensor<T>::grad() const {
        return grad_.get();
    }


    // Comparison operators
    template <typename T>
    bool Tensor<T>::shapeEqual (const Tensor<T>& other) const {
        return shape() == other.shape();
    }

    template <typename T>
    bool Tensor<T>::operator == (const Tensor<T>& other) const {
        std::vector<T> other_data = *other.data();
        return shapeEqual(other) && (*data() == other_data);
    }


    // Common tensors
    template <typename T>
    Tensor<T>& zeros(const std::vector<size_t> shape) {
        std::vector<T> zero_vector(numEntries(shape), 0);
        return *(new Tensor(zero_vector, shape)); 
    }

    template <typename T>
    Tensor<T>& ones(const std::vector<size_t> shape) {
        std::vector<T> zero_vector(numEntries(shape), 1);
        return *(new Tensor(zero_vector, shape)); 
    }


    // Prints 1D and 2D tensors
    template<typename U>
    std::ostream& operator << (std::ostream& out, const Tensor<U>& tensor){
        if(tensor.shape_.size() == 1){
            for(U entry : *tensor.data()){
                out << entry << " ";
            }
            out << std::endl;
        }else if(tensor.shape_.size() == 2){
            size_t n = tensor.shape_[0], m = tensor.shape_[1];
            for(size_t i = 0; i < n; ++i){
                for(size_t j = 0; j < m; ++j){
                    out << (*tensor.data())[i * m + j] << " ";
                }
                out << std::endl;
            }
        }
        return out;
    }

}
