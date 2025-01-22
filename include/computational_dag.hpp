#pragma once

#include "graph.hpp"

namespace tinytorch {

    /**
     * 
     * Functor-style class for tensor operations.
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct TensorOperation {
        std::function<Tensor<T>(const std::vector<const Tensor<T>&>)> tensorOperation_; 
        Tensor<T> tensorOperation(const std::vector<const Tensor<T>&> arguments);
    };

    /**
     * 
     * Node data of computational DAG nodes.
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct NodeData {
        TensorOperation<T> tensorOperation_; 
        Tensor<T> result;
    };

    /**
     * 
     * Class for (connected) computational DAGs with a single entry point (node with in-degree 0) and a single exit point (node with out-degree 0).
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template<typename T>
    class ComputationalDAG : Graph<NodeData<T>>{

            NodeId entry_point_;
            NodeId exit_point_;

        public:
            // TO DO: implement functions to check properties (acyclic, single entry, single exit, no multiple edges, ...)
            using Graph<NodeData<T>>::Graph; // Inherit all constructors from Graph
    };

} // namespace tinytorch
