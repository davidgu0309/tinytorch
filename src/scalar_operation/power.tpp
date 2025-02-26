namespace tinytorch{

    template<typename T>
    ScalarPow<T>::ScalarPow(){}
    
    template<typename T>
    T ScalarPow<T>::operator()(const std::vector<T> operands) const {
        return pow(operands[0], operands[1]);
    }
    
    template<typename T>
    T ScalarPow<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
        T base = operands[0], x = operands[1];
        if(input_idx) return log(base) * pow(base, x);
        return pow(base, x) / base * x;
    }
    
}  // namespace tinytorch