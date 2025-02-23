namespace tinytorch{

template <typename T>
Tensor<T> Matmul<T>::operator()(const std::vector<Tensor<T>> operands) const {
    assert(operands.size() == 2);
    return matmul<T>(operands[0], operands[1]);
}

template <typename T>
Tensor<T> Matmul<T>::backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const {
    Shape operand_shape = operands[input_idx].shape();
    Shape result_shape = matmulShape(operands[0].shape(), operands[1].shape());
    Shape jacobi_shape = concatIndexes(operand_shape, result_shape);
    Tensor<T> jacobi = zeros<T>(jacobi_shape);
    std::vector<MultiIndex> operand_indices = indexesRowMajor(operand_shape);
    std::vector<MultiIndex> result_indices = indexesRowMajor(result_shape);

    // {i1, i2, i3, i4} = 
    // i = {i1, i2}, k = {i3, i4}
    // 
    if (input_idx == 0) {
        for (MultiIndex i: operand_indices) {
            for (MultiIndex k: result_indices) {
                MultiIndex combined_index = concatIndexes(i, k);
                MultiIndex j = {i.back()};
                size_t rightOperandDim = operands[1].shape().size();
                for (size_t d=k.size()-rightOperandDim+1; d<k.size(); d++) {
                    j.push_back(k[d]);
                }
                // std::cout << i << std::endl;
                // std::cout << j << std::endl;
                // std::cout << k << std::endl;
                // std::cout << "-----" << std::endl;
                jacobi.getEntrySafe(combined_index) = (!k.size() || i.front() == k.front()) ? operands[1].getEntrySafe(j) : 0; 
            }
        }
    }
    else {
        for (MultiIndex j: operand_indices) {
            for (MultiIndex k: result_indices) {
                MultiIndex combined_index = concatIndexes(j, k);
                size_t leftOperandDim = operands[0].shape().size();
                MultiIndex i;
                for (size_t d=0; d<leftOperandDim-1; d++) {
                    i.push_back(k[d]);
                }
                i.push_back(j.front());
                jacobi.getEntrySafe(combined_index) = (!k.size() || j.back() == k.back()) ? operands[0].getEntrySafe(i) : 0; 
            }
        }
    }
    return jacobi;
    
}

} // namespace tinytorch