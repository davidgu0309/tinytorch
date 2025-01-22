#pragma once

#include "layer.hpp"
#include "dag.hpp"

namespace tinytorch {

// template <typename T>
// class Sequential {
//     public:
//         Sequential(std::vector<Layer<T>> layerList);

//         forward(const Tensor<T>& input); 
//         // {
//         //     for (Layer<T> L: layerList) {
//         //         input = L.forward(input);
//         //     }
//         // }
// };

template <typename T>
class Model {
    public:
        virtual Tensor<T> forward(const Tensor<T>& input) = 0; 

};

template <typename T>
struct TensorOperation {
    std::function<Tensor<T>(vector<Tensor<T>&>)> tensorOperation_; 
    Tensor<T> tensorOperation(vector<Tensor<T>& arguments);
};

template <typename T>
struct NodeData {
    TensorOperation tensorOperation_; 
    Tensor<T> result;
};

template <typename T>
class DAGModel {
    public:
        Tensor<T> forward(const Tensor<T>& input); 
    
    private:
        DAGNode<T> dagEntryPoint;

        
};

}

#include "../src/module.tpp"

