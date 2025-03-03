namespace tinytorch{

    template<typename T>
    ScalarSgn<T>::ScalarSgn(){}
    
    template<typename T>
    T ScalarSgn<T>::operator()(const std::vector<T> operands) const {
        if(operands[0] < 0) return -1;
        if(operands[0] > 0) return 1;
        return 0;
    }
    
    template<typename T>
    T ScalarSgn<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return (operands[0] == 0) ? std::numeric_limits<T>::infinity() : 0;
    }
    
    template<typename T>
    ScalarAbs<T>::ScalarAbs(){}
    
    template<typename T>
    T ScalarAbs<T>::operator()(const std::vector<T> operands) const {
        return abs(operands[0]);
    }
    
    template<typename T>
    T ScalarAbs<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        if(operands[0] < 0) return -1;
        if(operands[0] > 0) return 1;
        return 0;
    }

    template<typename T>
    ScalarNeg<T>::ScalarNeg(){}
    
    template<typename T>
    T ScalarNeg<T>::operator()(const std::vector<T> operands) const {
        return -operands[0];
    }
    
    template<typename T>
    T ScalarNeg<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        return -1;
    }
    
}  // namespace tinytorch