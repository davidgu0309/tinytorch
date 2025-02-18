namespace tinytorch{

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(){};

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(std::function<Tensor<T>(const std::vector<Tensor<T>>&)> tensorOperation) : tensorOperation_(tensorOperation) {};

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
std::vector<Tensor<T>> ComputationalDAG<T>::collectOperands(graph::NodeId node_id) const {
    std::vector<Tensor<T>> operands;
    if(id == entry_point_){
        operands.push_back(input);
    }else{
        for(const graph::NodeId operand_id : getPredecessors(id)){
            operands.push_back(get(operand_id).result_);
        }
    }
    return operands;
}

template<typename T>
Tensor<T> ComputationalDAG<T>::evaluate(const Tensor<T>& input) {
    const std::vector<graph::NodeId>& node_ids = topoOrder();
    for(const graph::NodeId id : node_ids){
        ComputationalDAGNode<T>& node = get(id);
        std::vector<Tensor<T>> operands = collectOperands(id);
        node.result_ = node.tensorOperation_(operands);
    }
    return get(exit_point_).result_;
}

template<typename T>
void ComputationalDAG<T>::backward(const Tensor<T>& input) {
    std::vector<graph::NodeId> node_ids = topoOrder();
    std::reverse(node_ids.begin(), node_ids.end());

    for(const graph::NodeId id : node_ids){
        ComputationalDAGNode<T>& node = get(id);
        //collect operands
        std::vector<Tensor<T>> operands = collectOperands(id);

        for(const graph::NodeId successor_id : getSuccessors(id)){
            for(size_t i=0; i<operands.size(); i++){ 
                gradient_wrt_operands[i] += matmul<T>(node.tensorOperation_.backward_wrt_operands(operands, i), get(successor_id).gradient_wrt_operands[operand_idx[id]]); 
            }
            for(size_t i=0; i<weights.size(); i++){ 
                gradient_wrt_weights[i] += matmul<T>(node.tensorOperation_.backward_wrt_weights(operands, i), get(successor_id).gradient_wrt_operands[operand_idx[id]]); 
            }
        }

        node.result_ = node.tensorOperation_(operands);
    }
    return get(exit_point_).result_;
}

}