namespace tinytorch{

    template<typename T>
    ScalarReLU<T>::ScalarReLU(){}
    
    template<typename T>
    T ScalarReLU<T>::operator()(const std::vector<T> operands) const {
        return (operands[0] > 0) ? operands[0] : 0;
    }
    
    template<typename T>
    T ScalarReLU<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return (operands[0] > 0) ? 1 : 0;
    }
    
    template<typename T>
    ScalarSigmoid<T>::ScalarSigmoid(){}
    
    template<typename T>
    T ScalarSigmoid<T>::operator()(const std::vector<T> operands) const {
        return (T)1 / (1 + std::exp(-operands[0]));
    }
    
    template<typename T>
    T ScalarSigmoid<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        T sig = (*this)(operands);
        return sig * (1 - sig);
    }
    
}  // namespace tinytorch