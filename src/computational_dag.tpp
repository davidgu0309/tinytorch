namespace tinytorch{

template<typename T>
const std::vector<NodeId>& ComputationalDAG<T>::topoOrder(){
    if (!is_topo_order_up_to_date){
        // Compute topo_order_
        topo_order_.clear();
        std::stack<NodeId> stack;
        std::vector<bool> visited(Graph<T>::size(), false);
        stack.push(entry_point_);
        visited[entry_point_] = true;
        while (!stack.empty()) {
            NodeId current = stack.top();
            stack.pop();
            topo_order_.push_back(current);
            for (NodeId neighbor : Graph<T>::adjacency_lists_[current]) {
                if (!visited[neighbor]) {
                    stack.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }
        std::reverse(topo_order_.begin(), topo_order_.end());
        is_topo_order_up_to_date = true;
    }
    return topo_order_;
}

template<typename T>
Tensor<T> ComputationalDAG<T>::evaluate(const Tensor<T>& input) {
    topoOrder();
    for(const NodeId id : topo_order_){
        ComputationalDAGNode<T> node = Graph<T>::get(id);
        std::vector<const Tensor<T>&> operands;
        if(id == entry_point_){
            operands.push_back(input);
        }else{
            for(const NodeId operand_id : Graph<T>::getPredecessors(id)){
                operands.push_back(Graph<T>::get(operand_id).result);
            }
            
        }
        node.result = node.tensorOperation(operands);
    }
    return Graph<T>::get(exit_point_).result;
}

}