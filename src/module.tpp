// def forward(x):
        // x1 = layer1(x)
        // x2 = x1 + x
        // x3 = layer2(x2)
        // return self.sigmoid(x3)

// class Layer:
//     ... 
//     def forward(x):
//             return self.weights * x + self.bias
    
// Node(x) Node(layer1.weight) Node(layer1.bias)
// Node (x1) Node(layer2.weight) Node(layer2.bias)
// Node (x2)
// Node (output)
// Node (loss)
template <typename T>
struct TensorOperation {
    std::function<Tensor<T>(vector<Tensor<T>&>)> tensorOperation_; 
    Tensor<T> tensorOperation(vector<Tensor<T>& arguments) {
        return tensorOperation_(arguments);
    }
};

template <typename T>
class DAGModel {
    public:
        Tensor<T> forward(const Tensor<T>& input) {
        
        }

        void topoSort() {
            if (is_topo_order_up_to_date) return;
            topo_order_.clear();
            std::vector<size_t> stack;
            std::vector<bool> visited(dag_.nodes_.size(), false);
            stack.push_back(dag.entry_point_);
            visited[dag.entry_point_] = true;

            while (!stack.empty()) {
                size_t current = stack.back();
                stack.pop_back();
                topo_order_.push_back(current);
                for (size_t neighbor : adjacency_list) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        stack.push_back(neighbor);
                    }
                }
            }
            std::reverse(topo_order_.begin(), topo_order_.end());
            is_topo_order_up_to_date = true;
        }
    
    private:
        DAG<NodeData<T>> dag_; //assume this dagEntryPoint is a correct entry point of an actual DAG
        std::vector<size_t> topo_order_;
        bool is_topo_order_up_to_date;


        
};