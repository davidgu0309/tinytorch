namespace tinytorch{

    template<typename T>
    ScalarSin<T>::ScalarSin(){}
    
    template<typename T>
    T ScalarSin<T>::operator()(const std::vector<T> operands) const {
        return sin(operands[0]);
    }
    
    template<typename T>
    T ScalarSin<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return cos(operands[0]);
    }
    
    template<typename T>
    ScalarCos<T>::ScalarCos(){}
    
    template<typename T>
    T ScalarCos<T>::operator()(const std::vector<T> operands) const {
        return cos(operands[0]);
    }
    
    template<typename T>
    T ScalarCos<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return -sin(operands[0]);
    }

    template<typename T>
    ScalarTan<T>::ScalarTan(){}
    
    template<typename T>
    T ScalarTan<T>::operator()(const std::vector<T> operands) const {
        return tan(operands[0]);
    }
    
    template<typename T>
    T ScalarTan<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        T cos_x = cos(x);
        return 1 / cos_x / cos_x;
    }
    
}  // namespace tinytorch