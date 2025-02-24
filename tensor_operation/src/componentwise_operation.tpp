template <typename T, ScalarOperation<T> ScalarOperation>
ComponentwiseOperation<T, ScalarOperation>::ComponentwiseOperation(){}

template <typename T, ScalarOperation<T> ScalarOperation>
Tensor<T> ComponentwiseOperation<T, ScalarOperation>::operator()(const std::vector<Tensor<T>> operands) const {
    assert(operands.size());
    Shape shape = operands[0].shape();
    Tensor<T> result(shape);
    size_t n = result.data().size(), m = operands.size();
    for(size_t i = 0; i < n; ++i){
        std::vector<T> scalar_operands;
        for(size_t j = 0; j < m; ++j){
            scalar_operands.push_back(operands[j].data()[i]);
        }
        result.data()[i] = scalarOperation_(scalar_operands);
    }
    return result;
}

template <typename T, ScalarOperation<T> ScalarOperation>
Tensor<T> ComponentwiseOperation<T, ScalarOperation>::backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const {
    Shape operand_shape = operands[input_idx].shape();
    Shape shape = concatIndexes(operand_shape, operand_shape);
    Tensor<T> jacobi = zeros<T>(shape);
    std::vector<MultiIndex> indices = indexesRowMajor(operand_shape);
    size_t m = operands.size();
    for (MultiIndex i: indices) {
        std::vector<T> scalar_operands;
        for(size_t j = 0; j < m; ++j){
            scalar_operands.push_back(operands[j].getEntrySafe(i));
        }
        MultiIndex combined_index = concatIndexes(i, i);
        jacobi.getEntrySafe(combined_index) = scalarOperation_.backward(input_idx, scalar_operands);
    }
    return jacobi;
}