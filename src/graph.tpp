template<typename T>
Graph<T>::Graph() {}

template<typename T>
Graph<T>::Graph(const size_t n) : nodes_(std::vector<T>(n)), adjacency_lists_(std::vector<std::vector<NodeId>>(n)) {}

template<typename T>
Graph<T>::Graph(std::vector<T> node_data, std::vector<std::vector<NodeId>> adjacency_lists) : nodes_(node_data), adjacency_lists_(adjacency_lists) {}

template<typename T>
NodeId Graph<T>::addNode(T node){
    NodeId new_node_id = nodes_.size();
    nodes_.push_back(node);
    adjacency_lists_.push_back({});
    return new_node_id;
}

template<typename T>
void Graph<T>::addEdge(NodeId from, NodeId to){
    adjacency_lists_[from].push_back(to);
}