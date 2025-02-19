namespace tinytorch{

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(){};

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(TensorOperation<T>* tensor_operation, std::vector<graph::NodeId> operand_node_id) : tensorOperation_(tensor_operation), operand_node_id_(operand_node_id) {
    size_t idx = 0;
    for(const graph::NodeId id : operand_node_id) operand_idx_[id] = idx++;
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

template <typename T> 
std::vector<Tensor<T>> ComputationalDAG<T>::collectOperands(const graph::NodeId node_id, const Tensor<T>& input) const {
    std::vector<Tensor<T>> operands;
    if(node_id == entry_point_){
        operands.push_back(input);
    }else{
        const std::vector<graph::NodeId>& operand_ids = get(node_id).operand_node_id_;
        for(const graph::NodeId operand_id : operand_ids){
            operands.push_back(get(operand_id).result_);
        }
    }
    return operands;
}

template<typename T>
Tensor<T> ComputationalDAG<T>::forward(const Tensor<T>& input) {
    const std::vector<graph::NodeId>& node_ids = topoOrder();
    for(const graph::NodeId id : node_ids){
        ComputationalDAGNode<T>& node = get(id);
        std::vector<Tensor<T>> operands = collectOperands(id, input);
        node.result_ = (*node.tensorOperation_)(operands);
    }
    return get(exit_point_).result_;
}

template<typename T>
void ComputationalDAG<T>::backward(const Tensor<T>& input) {
    std::vector<graph::NodeId> node_ids = topoOrder();
    std::reverse(node_ids.begin(), node_ids.end());
    for(const graph::NodeId id : node_ids){
        ComputationalDAGNode<T>& node = get(id);
        std::vector<Tensor<T>> operands = collectOperands(id, input);
        for(const graph::NodeId successor_id : getSuccessors(id)){
            ComputationalDAGNode<T>& successor = get(successor_id);
            for(size_t i=0; i<operands.size(); i++){ 
                node.gradients_wrt_inputs_[i] += matmul<T>(node->tensorOperation_.backwardWRTInputs(operands, i), successor.gradients_wrt_inputs_[successor.operand_idx_[id]]); 
            }
            for(size_t i=0; i<node.parameters_.size(); i++){ 
                node.gradients_wrt_parameters_[i] += matmul<T>(node->tensorOperation_.backwardWRTParameters(operands, i), successor.gradients_wrt_inputs_[successor.operand_idx_[id]]); 
            }
        }
    }
}

}