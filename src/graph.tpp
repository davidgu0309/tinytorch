template<typename T>
Graph<T>::Graph() {}

template<typename T>
Graph<T>::Graph(const size_t n) : nodes_(std::vector<T>(n)), adjacency_lists_(std::vector<std::vector<NodeId>>(n)) {}

template<typename T>
Graph<T>::Graph(std::vector<T> node_data, std::vector<std::vector<NodeId>> adjacency_lists) : nodes_(node_data), adjacency_lists_(adjacency_lists) {
    size_t n = node_data.size();
    assert(n == adjacency_lists.size());
    backward_adjacency_lists_ = std::vector<std::vector<NodeId>>(n, {});
    for(NodeId id = 0; id < n; ++id){
        for(NodeId succ_id : adjacency_lists[id]){
            backward_adjacency_lists_[succ_id].push_back(id);
        }
    }
}

template<typename T>
NodeId Graph<T>::size() const {
    return nodes_.size();
}

template<typename T>
T& Graph<T>::get(const NodeId id){
    return nodes_[id];
}

template<typename T>
const T& Graph<T>::get(const NodeId id) const {
    return nodes_[id];
}

template<typename T>
const std::vector<NodeId>& Graph<T>::getPredecessors(const NodeId id) const {
    return backward_adjacency_lists_[id];
}

template<typename T>
const std::vector<NodeId>& Graph<T>::getSuccessors(const NodeId id) const {
    return adjacency_lists_[id];
}

template<typename T>
NodeId Graph<T>::addNode(T node){
    NodeId new_node_id = nodes_.size();
    nodes_.push_back(node);
    adjacency_lists_.push_back({});
    backward_adjacency_lists_.push_back({});
    return new_node_id;
}

template<typename T>
void Graph<T>::addEdge(NodeId from, NodeId to){
    adjacency_lists_[from].push_back(to);
    backward_adjacency_lists_[to].push_back(from);
}