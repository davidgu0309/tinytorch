namespace tinytorch{

    template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(){};

template<typename T>
ComputationalDAGNode<T>::ComputationalDAGNode(std::function<Tensor<T>(const std::vector<Tensor<T>>&)> tensorOperation) : tensorOperation_(tensorOperation) {};

template<typename T>
ComputationalDAG<T>::ComputationalDAG() : graph::Graph<ComputationalDAGNode<T>>::Graph(), is_topo_order_up_to_date_(false){}

template<typename T>
ComputationalDAG<T>::ComputationalDAG(size_t n) : graph::Graph<ComputationalDAGNode<T>>::Graph(n), is_topo_order_up_to_date_(false){}

template<typename T>
ComputationalDAG<T>::ComputationalDAG(std::vector<T> node_data, std::vector<std::vector<graph::NodeId>> adjacency_lists) : graph::Graph<ComputationalDAGNode<T>>::Graph(node_data, adjacency_lists), is_topo_order_up_to_date_(false){}

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
graph::NodeId ComputationalDAG<T>::addNode(ComputationalDAGNode<int> node_data) {
    is_topo_order_up_to_date_ = false;
    return graph::Graph<ComputationalDAGNode<T>>::addNode(node_data);
}

template<typename T>
void ComputationalDAG<T>::addEdge(const graph::NodeId from, const graph::NodeId to){
    is_topo_order_up_to_date_ = false;
    graph::Graph<ComputationalDAGNode<T>>::addEdge(from, to);
}

template<typename T>
void ComputationalDAG<T>::DFS(graph::NodeId id, std::vector<bool>& visited){
    visited[id] = true;
    for(const graph::NodeId successor : getSuccessors(id)){
        if(!visited[successor]) DFS(successor, visited);
    }
    topo_order_.push_back(id);
}

template<typename T>
const std::vector<graph::NodeId>& ComputationalDAG<T>::topoOrder(){
    if (!is_topo_order_up_to_date_){
        // (Re)compute topo_order_
        topo_order_.clear();
        std::vector<bool> visited(size(), false);
        DFS(entry_point_, visited);
        std::reverse(topo_order_.begin(), topo_order_.end());
        is_topo_order_up_to_date_ = true;
    }
    return topo_order_;
}

template<typename T>
Tensor<T> ComputationalDAG<T>::evaluate(const Tensor<T>& input) {
    topoOrder();
    for(const graph::NodeId id : topo_order_){
        ComputationalDAGNode<T>& node = get(id);
        std::vector<Tensor<T>> operands;
        if(id == entry_point_){
            operands.push_back(input);
        }else{
            for(const graph::NodeId operand_id : getPredecessors(id)){
                operands.push_back(get(operand_id).result_);
            }
        }
        node.result_ = node.tensorOperation_(operands);
    }
    return get(exit_point_).result_;
}

}