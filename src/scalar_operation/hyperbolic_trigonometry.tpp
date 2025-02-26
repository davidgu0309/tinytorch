namespace tinytorch{

    template<typename T>
    ScalarSinH<T>::ScalarSinH(){}
    
    template<typename T>
    T ScalarSinH<T>::operator()(const std::vector<T> operands) const {
        return sinh(operands[0]);
    }
    
    template<typename T>
    T ScalarSinH<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return cosh(operands[0]);
    }
    
    template<typename T>
    ScalarCosH<T>::ScalarCosH(){}
    
    template<typename T>
    T ScalarCosH<T>::operator()(const std::vector<T> operands) const {
        return cosh(operands[0]);
    }
    
    template<typename T>
    T ScalarCosH<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return sinh(operands[0]);
    }

    template<typename T>
    ScalarTanH<T>::ScalarTanH(){}
    
    template<typename T>
    T ScalarTanH<T>::operator()(const std::vector<T> operands) const {
        return tanh(operands[0]);
    }
    
    template<typename T>
    T ScalarTanH<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        T tanh_x = tanh(x);
        return 1 - tanh_x * tanh_x;
    }
    
}  // namespace tinytorch