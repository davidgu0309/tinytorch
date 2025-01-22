template <typename T>
struct DAGNode {
    T data_;
    std::vector<DAGNode<T>&> next_list_;
    std::vector<DAGNode<T>&> prev_list_;

    DAGNode(T data) : data_(data) {};
};

template <typename T>
struct DAG {
    size_t entry_point_;
    std::vector<T> nodes_;
    std::vector<std::vector<size_t>> adjacency_list_;
};

