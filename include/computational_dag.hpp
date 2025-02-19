/**
 * @file computational_dag.hpp
 * 
 * @brief Template for tensor operation DAG data structure.
 * 
 * @author David Gu
 * @author Mirco Paul
 * 
 * @date \today
 */
#pragma once

#include "../DAG/dag.hpp"
#include "functional.hpp"
#include "tensor.hpp"
#include "tensor_operation.hpp"

#include <algorithm>
#include <functional>
#include <map>

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch {
    

    /**
     * @struct ComputationalDAGNode
     * 
     * @brief Structure for computational DAG nodes.
     * 
     * A computational DAG node represents a tensor operation. An instance of this
     * structure contains everything necessary for the forward and backward computations
     * of the corresponding node in the computational DAG.
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template <typename T>
    struct ComputationalDAGNode {
        TensorOperation<T>* tensorOperation_; /** Forward tensor operation. */
        Tensor<T> result_;  /** Result of forward computation. */
        std::vector<Tensor<T>> gradients_wrt_parameters_;
        std::vector<Tensor<T>> gradients_wrt_inputs_;
        std::vector<graph::NodeId> operand_node_id_; /** Stores the node id for each operand */
        std::map<graph::NodeId, size_t> operand_idx_; /** Inverse lookup of the above */

        // bool requires_grad_;

        ComputationalDAGNode();
        ComputationalDAGNode(TensorOperation<T>* tensor_operation, std::vector<graph::NodeId> operand_node_id);
    };

    /**
     * @class ComputationalDAG
     * 
     * @brief Acyclic tensor operation graph.
     * 
     * Class for (connected) computational DAGs with a single entry point (node with in-degree 0) and a single exit point (node with out-degree 0).
     * // @todo implement functions to check properties (acyclic, single entry, single exit, no multiple edges, ...)
     * 
     * @tparam T Floating point data type for numerical computations.
     * 
     **/
    template<typename T>
    class ComputationalDAG : dag::DAG<ComputationalDAGNode<T>>{

            graph::NodeId entry_point_;    /** This node acts directly on the input to the DAG. */
            graph::NodeId exit_point_;     /** The result of this node is the result of the DAG computation. */

        public:

            /**
             * 
             * Constructs an empty computational graph.
             * 
             **/
            ComputationalDAG();

            /**
             * 
             * Constructs a computational graph with n nodes, no edges and uninitialized node data.
             * 
             **/
            ComputationalDAG(size_t n);
            
            /**
             * 
             * Assumes node_data.size() == adjacency_lists.size() and adjacency_lists valid DAG 
             * that satisfies specification properties.
             * 
             **/
            ComputationalDAG(std::vector<T> node_data, std::vector<std::vector<graph::NodeId>> adjacency_lists);

            // TO DO: add constructor with entry and exit

            /**
             * 
             * @return Number of nodes in the graph.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::size;

            /**
             * 
             * @return Reference to the entry point identifier.
             * 
             **/
            graph::NodeId& getEntryPoint();

            /**
             * 
             * @return Immutable reference to the entry point identifier.
             * 
             **/
            const graph::NodeId& getEntryPoint() const;

            /**
             * 
             * @return Reference to the exit point identifier.
             * 
             **/
            graph::NodeId& getExitPoint();

            /**
             * 
             * @return Immutable reference to the exit point identifier.
             * 
             **/
            const graph::NodeId& getExitPoint() const;

            /**
             * 
             * @param id Node identifier.
             * 
             * @return Data of node with identifier id.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::get;

            /**
             * 
             * @param id Node identifier.
             * 
             * @return Immutable reference to vector of predecessors of node with identifier id.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::getPredecessors;

            /**
             * 
             * @param id Node identifier.
             * 
             * @return Immutable reference to vector of successors of node with identifier id.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::getSuccessors;

            /**
             * 
             * Adds a node with data node_data to the graph.
             * @param node_data Node data.
             * 
             * @return Identifier of the new node.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::addNode;

            /**
             * 
             * Adds a directed edge from node from to node to.
             * @param from Source node.
             * 
             * @param to Destination node.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::addEdge;


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
            using dag::DAG<ComputationalDAGNode<T>>::topoOrder;

            std::vector<Tensor<T>> collectOperands(const graph::NodeId node_id, const Tensor<T>& input) const;

            // ASSUMES ORDER OF OPERANDS EQUAL TO ORDER OF EDGES IN BACKWARD ADJACENCY LIST
            Tensor<T> forward(const Tensor<T>& input);

            // ASSUMES EVALUATE HAS BEEN CALLED
            void backward(const Tensor<T>& input);

    };

} // namespace tinytorch

#include "../src/computational_dag.tpp"
