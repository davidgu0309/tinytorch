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

namespace tinytorch{

template <typename T>
Tensor<T> DAGModel<T>::forward(const Tensor<T>& input) {
    computational_graph_.evaluate(input);
}

}