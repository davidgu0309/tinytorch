#pragma once

#include <stdlib.h>
#include <vector>

/**
 * 
 * Type for node identifiers. NodeId id is the identifier for the node in Graph::nodes_[id].
 * 
 **/
typedef size_t NodeId;

/**
 * 
 * Adjacency-lists based class for directed (multi-)graphs.
 * @tparam T Node type.
 * 
 **/
template <typename T>
class Graph {

    std::vector<T> nodes_;
    std::vector<std::vector<NodeId>> adjacency_lists_;
    std::vector<std::vector<NodeId>> backward_adjacency_lists_;

    public:
        /**
         * 
         * Constructs an empty graph.
         * 
         **/
        Graph();

        /**
         * 
         * Constructs a graph with n nodes, no edges and empty node data.
         * 
         **/
        Graph(const size_t n);

        /**
         * 
         * Obvious.
         * 
         **/
        Graph(std::vector<T> node_data, std::vector<std::vector<NodeId>> adjacency_lists);
        
        /**
         * 
         * @return Number of nodes in the graph.
         * 
         **/
        size_t size() const;

        /**
         * 
         * @param id Node identifier.
         * 
         * @return Data of node with identifier id.
         * 
         **/
        T& get(const NodeId id);

        /**
         * 
         * @param id Node identifier.
         * 
         * @return Data of node with identifier id.
         * 
         **/
        const T& get(const NodeId id) const;

        /**
         * 
         * @param id Node identifier.
         * 
         * @return Immutable reference to vector of predecessors of node with identifier id.
         * 
         **/
        const std::vector<NodeId>& getPredecessors(const NodeId id) const;

        /**
         * 
         * @param id Node identifier.
         * 
         * @return Immutable reference to vector of successors of node with identifier id.
         * 
         **/
        const std::vector<NodeId>& getSuccessors(const NodeId id) const;

        /**
         * 
         * Adds a node with data node_data to the graph.
         * @param node_data Node data.
         * 
         * @return Identifier of the new node.
         * 
         **/
        NodeId addNode(T node_data);

        /**
         * 
         * Adds a directed edge from node from to node to.
         * @param from Source node.
         * 
         * @param to Destination node.
         * 
         **/
        void addEdge(const NodeId from, const NodeId to);
};

#include "../src/graph.tpp"

/* // "Linked list style", unused for now
template <typename T>
struct GraphNode {
    T data_;
    std::vector<DAGNode<T>&> next_list_;
    std::vector<DAGNode<T>&> prev_list_;

    DAGNode(T data) : data_(data) {};
};
*/