namespace tinytorch{

    template<typename T>
    ScalarLog<T>::ScalarLog(){}
    
    template<typename T>
    T ScalarLog<T>::operator()(const std::vector<T> operands) const {
        return log(operands[1]) / log(operands[0]);
    }
    
    template<typename T>
    T ScalarLog<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        T base = operands[0], x = operands[1];
        if(input_idx) return 1 / log(b) / x;
        T log_x = log(x);
        return -log(x) / b / log_x / log_x;
    }
    
}  // namespace tinytorch