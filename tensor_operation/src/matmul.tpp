namespace tinytorch{

template <typename T>
Tensor<T> Matmul<T>::operator()(const std::vector<Tensor<T>> operands) const {
    assert(operands.size() == 2);
    return matmul<T>(operands[0], operands[1]);
}

template <typename T>
Tensor<T> Matmul<T>::backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const {
    Shape operand_shape = operands[input_idx].shape();
    Shape jacobi_shape = operand_shape;
    Shape result_shape = matmulShape(operands[0].shape(), operands[1].shape());
    jacobi_shape.insert(jacobi_shape.end(), result_shape.begin(), result_shape.end());
    Tensor<T> jacobi = zeros<T>(jacobi_shape);
    std::vector<MultiIndex> operand_indices = indexesRowMajor(operand_shape);
    std::vector<MultiIndex> result_indices = indexesRowMajor(result_shape);

    if (input_idx == 0) {
        for (MultiIndex i: operand_indices) {
            for (MultiIndex k: result_indices) {
                MultiIndex combined_index = concatIndexes(i, k);
                MultiIndex j = {i.back()};
                size_t rightOperandDim = operands[1].shape().size();
                for (size_t d=k.size()-rightOperandDim+1; d<k.size(); d++) {
                    j.push_back(k[d]);
                }
                i.pop_back();
                while (k.size() > i.size()) k.pop_back();
                jacobi.get(combined_index) = i == k ? operands[1].get(j) : 0; 
            }
        }
    }
    else {
        for (MultiIndex j: operand_indices) {
            for (MultiIndex k: result_indices) {
                MultiIndex combined_index = concatIndexes(j, k);
                MultiIndex i;
                size_t leftOperandDim = operands[0].shape().size();
                for (size_t d=0; d<leftOperandDim-1; d++) {
                    i.push_back(k[d]);
                }
                i.push_back(j.front());
                j.pop_back();
                while (k.size() > j.size()) k.pop_back();
                jacobi.get(combined_index) = j == k ? operands[0].get(i) : 0; 
            }
        }
    }
    
}

} // namespace tinytorch