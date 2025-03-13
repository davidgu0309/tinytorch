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
#include "../tensor/include/functional.hpp"
#include "../tensor/include/tensor.hpp"
#include "tensor_operation/tensor_operation.hpp"

#include <algorithm>
#include <functional>
#include <map>

using namespace tensor;

/**
 * @namespace tinytorch
 * 
 * @brief Namespace of the entire framework.
 * 
 */
namespace tinytorch {

    typedef size_t InputId;
    typedef size_t ParameterId;

    union OperandId{
        InputId input_id_;
        ParameterId parameter_id_;
        graph::NodeId node_id_;
    };

    enum OperandType{
        INPUT,
        PARAMETER,
        NODE
    };

    struct OperandDescriptor{

        OperandType operand_type_;
        OperandId id_;

        OperandDescriptor(size_t id, OperandType type);

        /** // Is there a trick to make this work?
        OperandDescriptor(InputId id);
        OperandDescriptor(ParameterId id);
        OperandDescriptor(graph::NodeId id);
        */
    };

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

        TensorOperation<T>& tensorOperation_; /** Forward tensor operation. */
        Tensor<T> result_;  /** Result of forward computation. */
        std::vector<Tensor<T>> jacobi_; /** jacobi_[i] is used to store the Jacobi tensor wrt to the i-th input.  */
        std::vector<Tensor<T>> jacobi_accumulator_; /** jacobi_accumulator_[i] stores the accumulated jacobi i over the data points of a batch  */

        std::vector<OperandDescriptor> operand_descriptor_; /** Stores the operand descriptor for each operand. */
        std::map<graph::NodeId, size_t> operand_idx_; /** Indexes of NODE operands in operand_descriptor_. */

        ComputationalDAGNode();
        ComputationalDAGNode(TensorOperation<T>& tensor_operation, std::vector<OperandDescriptor> operand_descriptors);
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

            std::vector<Tensor<T>> inputs_; /** Used to store the input values. */
            std::vector<Tensor<T>> parameters_; /** Used to store the parameters. */

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
             * Adds a node with data node_data to the graph. Adds edges to the graph based on operand descriptors.
             * 
             * @param dag_node Node to add.
             * 
             * @return Identifier of the new node.
             * 
             **/
            graph::NodeId addNode(ComputationalDAGNode<T> dag_node);

            /**
             * 
             * Adds a directed edge from node from to node to.
             * 
             * @param from Source node.
             * @param to Destination node.
             * 
             **/
            using dag::DAG<ComputationalDAGNode<T>>::addEdge;

            /**
             * 
             * Allocates an input tensor of shape shape and returns the unique InputId.
             * 
             * @param shape Shape of allocated input tensor.
             * 
             * @return Unique input identifier.
             * 
             **/
            InputId addInput(Shape shape);

            /**
             * 
             * Allocates a parameter tensor of shape shape and returns the unique ParameterId.
             * 
             * @param shape Shape of allocated parameter tensor.
             * 
             * @return Unique parameter identifier.
             * 
             **/
            InputId addParameter(Shape shape);

            /**
             * 
             * @param id Input identifier.
             * 
             * @return Const reference to input tensor with identifier id.
             * 
             **/
            const Tensor<T>& getInput(InputId id) const;

            /**
             * 
             * @param id Input identifier.
             * 
             * @return Reference to input tensor with identifier id.
             * 
             **/
            Tensor<T>& getInput(InputId id);

            /**
             * 
             * @param id Parameter identifier.
             * 
             * @return Const reference to parameter tensor with identifier id.
             * 
             **/
            const Tensor<T>& getParameter(ParameterId id) const;

            /**
             * 
             * @param id Parameter identifier.
             * 
             * @return Reference to parameter tensor with identifier id.
             * 
             **/
            Tensor<T>& getParameter(ParameterId id);


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

            std::vector<Tensor<T>> collectOperands(const graph::NodeId node_id) const;

            // ASSUMES INPUTS AND WEIGHTS HAVE BEEN SET
            Tensor<T> forward();

            // ASSUMES INPUTS AND WEIGHTS HAVE BEEN SET
            // ASSUMES EVALUATE HAS BEEN CALLED
            void backward();

    };

} // namespace tinytorch

#include "../src/computational_dag.tpp"
