namespace tinytorch{

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(){};

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(TensorOperation<T>& tensor_operation, std::vector<OperandDescriptor> operand_descriptors) : tensorOperation_(tensor_operation), operand_descriptor_(operand_descriptors) {
    size_t idx = 0;
    for(const OperandDescriptor descriptor : operand_descriptors){
        if(descriptor.operand_type_ == NODE) operand_idx_[descriptor.id_.node_id_] = idx;
        idx++;
    }
};

template<typename T>
ComputationalDAG<T>::ComputationalDAG() : dag::DAG<ComputationalDAGNode<T>>::DAG(){}

template<typename T>
ComputationalDAG<T>::ComputationalDAG(size_t n) : dag::DAG<ComputationalDAGNode<T>>::DAG(n){}

template<typename T>
ComputationalDAG<T>::ComputationalDAG(std::vector<T> node_data, std::vector<std::vector<graph::NodeId>> adjacency_lists) : dag::DAG<ComputationalDAGNode<T>>::DAG(node_data, adjacency_lists){}

template<typename T>
graph::NodeId& ComputationalDAG<T>::getEntryPoint(){
    return entry_point_;
}

template<typename T>
const graph::NodeId& ComputationalDAG<T>::getEntryPoint() const {
    return entry_point_;
}

template<typename T>
graph::NodeId& ComputationalDAG<T>::getExitPoint(){
    return exit_point_;
}

template<typename T>
const graph::NodeId& ComputationalDAG<T>::getExitPoint() const {
    return exit_point_;
}

template<typename T>
InputId ComputationalDAG<T>::addInput(Shape shape){
    inputs_.push_back(zeros<T>(shape));
    return inputs_.size() - 1;
}

template<typename T>
ParameterId ComputationalDAG<T>::addParameter(Shape shape){
    parameters_.push_back(zeros<T>(shape));
    return parameters_.size() - 1;
}

template<typename T>
const Tensor<T>& ComputationalDAG<T>::getInput(InputId id) const {
    return inputs_[id];
}

template<typename T>
Tensor<T>& ComputationalDAG<T>::getInput(InputId id){
    return inputs_[id];
}

template<typename T>
const Tensor<T>& ComputationalDAG<T>::getParameter(ParameterId id) const {
    return parameters_[id];
}

template<typename T>
Tensor<T>& ComputationalDAG<T>::getParameter(ParameterId id){
    return parameters_[id];
}

template <typename T> 
std::vector<Tensor<T>> ComputationalDAG<T>::collectOperands(const graph::NodeId node_id) const {
    std::vector<Tensor<T>> operands;
    const std::vector<graph::NodeId>& operand_descriptors = get(node_id).operand_descriptor_;
    for(const OperandDescriptor desc : operand_descriptors){
        if(desc.operand_type_ == NODE) operands.push_back(get(desc.id_.node_id_).result_);
        else if(desc.operand_type_ == INPUT) operands.push_back(inputs_[desc.id_.input_id_]);
        else operands.push_back(parameters_[desc.id_.parameter_id_]);
    }
    return operands;
}

template<typename T>
Tensor<T> ComputationalDAG<T>::forward() {
    const std::vector<graph::NodeId>& node_ids = topoOrder();
    for(const graph::NodeId id : node_ids){
        ComputationalDAGNode<T>& node = get(id);
        std::vector<Tensor<T>> operands = collectOperands(id);
        node.result_ = (*node.tensorOperation_)(operands);
    }
    return get(exit_point_).result_;
}

template<typename T>
void ComputationalDAG<T>::backward() {
    std::vector<graph::NodeId> node_ids = topoOrder();
    std::reverse(node_ids.begin(), node_ids.end());
    for(const graph::NodeId id : node_ids){
        ComputationalDAGNode<T>& node = get(id);
        std::vector<Tensor<T>> operands = collectOperands(id);
        for(const graph::NodeId successor_id : getSuccessors(id)){
            ComputationalDAGNode<T>& successor = get(successor_id);
            for(size_t i = 0; i < operands.size(); i++){ 
                node.jacobi_[i] += evaluateDifferential<T>(node->tensorOperation_.backward(operands, i), successor.jacobi_[successor.operand_idx_[id]]); 
            }
        }
    }
}

}