namespace tinytorch{

    template <typename T>
    Tensor<T> TensorAddition<T>::operator()(std::vector<Tensor<T>>& operands) const {
        assert(operands.size());
        Shape shape = operands[0].shape_;
        Tensor<T> result = zeros<T>(shape);
        for(const Tensor<T>& operand : operands){
            result = add<T>(result, operand);
        }
        return result;
    }

    template <typename T>
    Tensor<T> TensorAddition<T>::backwardWRTInputs(size_t input_idx, std::vector<Tensor<T>>& operands) const {
        Shape operand_shape = operands[input_idx].shape_;
        Shape shape = operand_shape;
        shape.insert(shape.end(), operand_shape.begin(), operand_shape.end());
        Tensor<T> jacobi = zeros<T>(shape);
        std::vector<MultiIndex> indices = indexesRowMajor(operand_shape);
        for (MultiIndex i: indices) {
            MultiIndex combined_index = combineIndexes(i, i);
            jacobi.getEntrySafe(combined_index) = 1;
        }
        return jacobi;
    }

    // a1 a2
    // a3 b4

    // +

    // b1 b2
    // b3 b4

    // = 

    // a1+b1 a2+b2
    // a3+b3 a4+b4 


    // w_updated = w - learning rate * Jacobian * x
    //                 learning rate * matrix full of 1s hadamard x

    template <typename T>
    Tensor<T> Matmul<T>::operator()(std::vector<Tensor<T>>& operands) const {
        assert(operands.size() == 2);
        return matmul<T>(operands[0], operands[1]);
    }

    template <typename T>
    Tensor<T> Matmul<T>::backwardWRTInputs(size_t input_idx, std::vector<Tensor<T>>& operands) const {
        Shape operand_shape = operands[input_idx].shape_;
        Shape jacobi_shape = operand_shape;
        Shape result_shape = matmulShape(operand[0].shape_, operand[1].shape_);
        jacobi_shape.insert(jacobi_shape.end(), result_shape.begin(), result_shape.end());
        Tensor<T> jacobi = zeros<T>(jacobi_shape);
        std::vector<MultiIndex> operand_indices = indexesRowMajor(operand_shape);
        std::vector<MultiIndex> result_indices = indexesRowMajor(result_shape);

        if (input_idx == 0) {
            for (MultiIndex i: operand_indices) {
                for (MultiIndex j: result_indices) {
                    MultiIndex combined_index = combineIndexes(i, j);
                    jacobi.get(combined_index) = i.back() == j.front() ? operand[1].get(j) : 0; 
                }
            }
        }
        else {
            for (MultiIndex i: operand_indices) {
                for (MultiIndex j: result_indices) {
                    MultiIndex combined_index = combineIndexes(i, j);
                    jacobi.get(combined_index) = i.back() == j.front() ? operand[1].get(j) : 0; 
                }
            }
        }
        
    }



    

} // namespace tinytorch

