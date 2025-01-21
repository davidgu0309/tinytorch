namespace tinytorch {

    bool multiIndexLegalityTest(const Shape shape, const MultiIndex index){
        if(shape.size() != index.size()) return false;
        for(int i = 0; i < shape.size(); ++i){
            if(index[i] >= shape[i]) return false;
        }
        return true;
    }

    std::ostream& operator << (std::ostream& out, const MultiIndex& index){
        for(size_t i : index){
            out << i << " ";
        }
        // out << std::endl;
        return out;
    }

    // Constructors
    template <typename T>
    Tensor<T>::Tensor(){}
    
    template <typename T>
    Tensor<T>::Tensor(const T value) : shape_({}), data_(std::vector<T>(1, value)){}

    template <typename T>
    Tensor<T>::Tensor(const Shape shape) : shape_(shape), data_(std::vector<T>(numEntries(shape))){}

    template <typename T>
    Tensor<T>::Tensor(const Shape shape, const std::vector<T>& data) : shape_(shape), data_(data){}


    template <typename T>
    size_t Tensor<T>::size() const {
        return data().size();
    }

    template <typename T>
    Shape Tensor<T>::shape() const {
        return shape_;
    }


    template <typename T>
    std::vector<T>& Tensor<T>::data() {
        return data_;
    }

    template <typename T>
    std::vector<T>& Tensor<T>::grad() {
        return grad_;
    }


    template <typename T>
    const std::vector<T>& Tensor<T>::data() const {
        return data_;
    }

    template <typename T>
    const std::vector<T>& Tensor<T>::grad() const {
        return grad_;
    }

    template <typename T>
    T& Tensor<T>::getEntryUnsafe(MultiIndex index){
        size_t data_pos = index.size() ? index[0] : 0;
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
        return data()[data_pos];
    }

    template <typename T>
    const T& Tensor<T>::getEntryUnsafe(MultiIndex index) const{
        size_t data_pos = index.size() ? index[0] : 0;
        for(size_t d = 1; d < index.size(); ++d){
            data_pos *= shape_[d];
            data_pos += index[d];
        }
        return data()[data_pos];
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
        const std::vector<T>& other_data = other.data();
        return shapeEqual(other) && (data() == other_data);
    }


    // Common tensors
    template <typename T>
    Tensor<T> constant(const std::vector<size_t> shape, T value){
        std::vector<T> data(numEntries(shape), value);
        return Tensor(shape, data);
    }

    template <typename T>
    Tensor<T> zeros(const Shape shape) {
        return constant<T>(shape, 0);
    }

    template <typename T>
    Tensor<T> ones(const Shape shape) {
        return constant<T>(shape, 1);
    }

    template <typename T>
    Tensor<T> iota(const std::vector<size_t> shape){
        std::vector<T> data(numEntries(shape));
        std::iota(data.begin(), data.end(), 1);
        return Tensor(shape, data);
    }

    template <typename T>
    Tensor<T> initialize_using_generator(const std::vector<size_t> shape, std::function<T()> generator) {
        std::vector<T> data(numEntries(shape));
        for (int i=0; i<data.size(); i++) {
            data[i] = generator();
        }
        return Tensor(shape, data);
    }

    template <typename T>
    Tensor<T> real_uniform(const std::vector<size_t> shape, const T lower, const T upper) {
        std::random_device rd;
        std::mt19937 gen(rd());
        return initialize_using_generator<T>(shape, [lower, upper, gen]() mutable {
            std::uniform_real_distribution<T> dist(lower, upper);
            return dist(gen);
        });
    }



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
        // Iterate over all indexes and print
        Shape shape = tensor.shape();
        // out << shape << std::endl;
        size_t n = shape.size(), m = tensor.size();
        for(size_t i = 0; i < m; ++i){
            out << tensor.data()[i] << " ";
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
