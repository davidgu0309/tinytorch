namespace tinytorch{

    template <typename T>
    Tensor<T> TensorAddition<T>::operator()(std::vector<Tensor<T>>& operands) const {
        assert(operands.size());
        Shape shape = operands[0].shape_;
        Tensor<T> result = zeros(shape);
        for(const Tensor<T>& operand : operands){
            result = add(result, operand);
        }
        return result;
    }

    

} // namespace tinytorch