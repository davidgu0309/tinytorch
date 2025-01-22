#include "../include/computational_dag.hpp"

#include <iostream>

namespace tinytorch {

ComputationalDAGNode<int> id_node, plus_2_node, times_3_node, sum_node;

void computationalDAGUnitTests(){
    // TO DO: improve framework and rewrite this
    ComputationalDAG<int> computational_dag;
    NodeId id_node_id = computational_dag.addNode(id_node);
    NodeId plus_2_node_id = computational_dag.addNode(plus_2_node);
    NodeId times_3_node_id = computational_dag.addNode(times_3_node);
    NodeId sum_node_id = computational_dag.addNode(sum_node);
    std::cout << "Size test 1: " << (computational_dag.size() == 4 ? "Passed" : "Failed") << std::endl;
}
    
} // namespace tinytorch
