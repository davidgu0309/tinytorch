namespace tinytorch {

    bool multiIndexLegalityTest(const Shape shape, const MultiIndex index){
        if(shape.size() != index.size()) return false;
        for(int i = 0; i < shape.size(); ++i){
            if(index[i] >= shape[i]) return false;
        }
        return true;
    }

    // Constructors
    template <typename T>
    Tensor<T>::Tensor(const Shape shape) : shape_(shape) {
        std::vector<T>* empty_data = new std::vector<T>(numEntries(shape));
        data_ = std::unique_ptr<std::vector<T>>(empty_data);
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<T>& data, 
        const Shape shape) : shape_(shape)
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
    Shape Tensor<T>::shape() const {
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

    template <typename T>
    T& Tensor<T>::getEntryUnsafe(MultiIndex index){
        size_t data_pos = index[0];
        // shape_ = {2, 2, 3}
        // (index[0] * shape_[1] + index[1]) * shape_[2] + index[2]
        // (index[0] * shape_[1] * shape_[2] + index[1] * shape_[2] + index[2])
        /*
        _ _ _
        _ _ _

        _ _ _
        _ _ _

        Horner Scheme

        sum_i a_i * x ** i = (a_i * x + a_(i - 1)) * x + ...

        */
        for(size_t d = 1; d < index.size(); ++d){
            data_pos *= shape_[d];
            data_pos += index[d];
        }
        return (*data())[data_pos];
    }

    template <typename T>
    const T& Tensor<T>::getEntryUnsafe(MultiIndex index) const{
        size_t data_pos = index[0];
        for(size_t d = 1; d < index.size(); ++d){
            data_pos *= shape_[d];
            data_pos += index[d];
        }
        return (*data())[data_pos];
    }

    template <typename T>
    T& Tensor<T>::getEntrySafe(MultiIndex index){
        assert(multiIndexLegalityTest(shape_, index));
        return getEntryUnsafe(index);
    }

    template <typename T>
    const T& Tensor<T>::getEntrySafe(MultiIndex index) const{
        assert(multiIndexLegalityTest(shape_, index));
        return getEntryUnsafe(index);
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
    Tensor<T>& zeros(const Shape shape) {
        std::vector<T> zero_vector(numEntries(shape), 0);
        return *(new Tensor(zero_vector, shape)); 
    }

    template <typename T>
    Tensor<T>& ones(const Shape shape) {
        std::vector<T> zero_vector(numEntries(shape), 1);
        return *(new Tensor(zero_vector, shape)); 
    }

    /*
    template <typename T>
    Tensor<T>& iota(const Shape shape) {
        // TO DO: implement iota, maybe recursively
    }*/


    /**
     * Writes tensor to output stream out. This enables std::cout << tensor ...
     *
     * @tparam U Tensor entry type.
     * 
     * @param out Output stream.
     * @param tensor Tensor to print.
     * 
     * @return Updated output stream.
     * 
     **/
    template<typename U>
    std::ostream& operator << (std::ostream& out, const Tensor<U>& tensor){
        /* // Old version: only works for 1D and 2D
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
        */

        /* // Generate multiindexes. COMPLETELY UNNECESSARY
        std::queue<MultiIndex> indexes; // Multiindexes in "row-major" order
        indexes.push({});
        Shape shape = tensor.shape();
        size_t n = shape.size();
        for(size_t d = 0; d < n; ++d){
            size_t m = indexes.size();
            // Iterate over all multiindexes of the previous dimension
            for(size_t i = 0; i < m; ++i){
                // For each one, add all possible indexes for the current dimension
                for(size_t j = 0; j < shape[d]; ++j){
                    MultiIndex index = indexes.front();
                    index.push_back(j);
                    indexes.push(index);
                }
                indexes.pop();
            }
        }
        */
        
        // Iterate over all indexes and print
        Shape shape = tensor.shape();
        size_t n = shape.size(), m = tensor.size();
        for(size_t i = 0; i < m; ++i){
            out << (*tensor.data())[i] << " ";
            // Inefficient but doesn't matter
            size_t j = 0, temp = i + 1;
            while(j < n && !(temp % shape[n - 1 - j])){
                out << std::endl;
                temp /= shape[n - 1 - j];
                ++j;
            }
        }
        
        return out;
    }

}
