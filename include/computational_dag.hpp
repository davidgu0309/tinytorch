#pragma once

#include "graph.hpp"

#include <algorithm>
#include <stack>

namespace tinytorch {

    /**
     * 
     * Functor-style class for tensor operations.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct TensorOperation {
        std::function<Tensor<T>(const std::vector<const Tensor<T>&>)> tensorOperation_; 
        Tensor<T> operator()(const std::vector<const Tensor<T>&> operands);
    };

    /**
     * 
     * Node data of computational DAG nodes.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct NodeData {
        TensorOperation<T> tensorOperation; 
        Tensor<T> result_;
    };

    /**
     * 
     * Class for (connected) computational DAGs with a single entry point (node with in-degree 0) and a single exit point (node with out-degree 0).
     * // TO DO: implement functions to check properties (acyclic, single entry, single exit, no multiple edges, ...)
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template<typename T>
    class ComputationalDAG : Graph<NodeData<T>>{

            NodeId entry_point_;
            NodeId exit_point_;

            bool is_topo_order_up_to_date;
            std::vector<size_t> topo_order_;

        public:
            using Graph<NodeData<T>>::Graph; // Inherit all constructors from Graph

            /**
             * 
             * Lazily computes a topological order of the nodes in the current DAG.
             * If the graph has been modified the topological order is recomputed.
             * If the graph satisfies the specifications the first node in any topological 
             * order is entry_point and the last node in any topological order is exit_point_.
             * 
             * TO DO: is there a way to keep the topological order up to date in a more efficient way?
             * 
             * @return A vector of node identifiers in topological order.
             * 
             **/
            const std::vector<NodeId>& topoOrder();

            // ASSUMES ORDER OF OPERANDS EQUAL TO ORDER OF EDGES IN BACKWARD ADJACENCY LIST
            Tensor<T> evaluate(const Tensor<T>& input);
    };

} // namespace tinytorch

#include "../src/computational_dag.tpp"
