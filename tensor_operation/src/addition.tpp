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

namespace tinytorch{

    template <typename T>
    Tensor<T> TensorAddition<T>::operator()(const std::vector<Tensor<T>> operands) const {
        assert(operands.size());
        Shape shape = operands[0].shape();
        Tensor<T> result = zeros<T>(shape);
        for(const Tensor<T>& operand : operands){
            result = add<T>(result, operand);
        }
        return result;
    }

    template <typename T>
    Tensor<T> TensorAddition<T>::backward(const size_t input_idx, const std::vector<Tensor<T>> operands) const {
        Shape operand_shape = operands[input_idx].shape();
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

} // namespace tinytorch

    



