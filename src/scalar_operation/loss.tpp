namespace tinytorch{

    template<typename T>
    ScalarLp<T>::ScalarLp(){}
    
    template<typename T>
    T ScalarLp<T>::operator()(const std::vector<T> operands) const {
        T d = operands[0] - operands[1];
        return d * d;
    }
    
    template<typename T>
    T ScalarLp<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        T d = operands[0] - operands[1];
        if(input_idx) return -2 * d;
        return 2 * d;
    }
    
}  // namespace tinytorch