template<typename T>
ScalarAddition<T>::ScalarAddition(){}

template<typename T>
T ScalarAddition<T>::operator()(const std::vector<T> operands) const {
    T sum = 0;
    for(const T operand : operands){
        sum += operand;
    }
    return sum;
}

template<typename T>
T ScalarAddition<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
    return 1;
}

template<typename T>
ScalarMultiplication<T>::ScalarMultiplication(){}

template<typename T>
T ScalarMultiplication<T>::operator()(const std::vector<T> operands) const {
    T product = 1;
    for(const T operand : operands){
        product *= operand;
    }
    return product;
}

template<typename T>
T ScalarMultiplication<T>::backward(const size_t input_idx, const std::vector<T> operands) const {
    T derivative = 1;
    for(size_t idx = 0; idx < operands.size(); ++idx){
        if(idx != input_idx) derivative *= operands[idx];
    }
    return derivative;
}