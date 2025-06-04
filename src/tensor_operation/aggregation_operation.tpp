namespace tinytorch{

template <typename T, ScalarOperation<T> ScalarOperation>
AggregationOperation<T, ScalarOperation>::AggregationOperation(){}

template <typename T, ScalarOperation<T> ScalarOperation>
Tensor<T> AggregationOperation<T, ScalarOperation>::operator()(const std::vector<Tensor<T>> operand) const {
    assert(operand.size() == 1);
    Tensor<T> result(scalarOperation_(operand[0].data()));
    return result;
}

template <typename T, ScalarOperation<T> ScalarOperation>
Tensor<T> AggregationOperation<T, ScalarOperation>::backward(const size_t input_idx, const std::vector<Tensor<T>> operand) const {
    assert(!input_idx);
    Shape shape = operand[0].shape();
    Tensor<T> jacobi = zeros<T>(shape);
    size_t n = operand[0].size();
    for(size_t i = 0; i < n; ++i){
        jacobi.data()[i] = scalarOperation_.backward(i, operand[0].data());
    }
    return jacobi;
}

}  // namespace tinytorch