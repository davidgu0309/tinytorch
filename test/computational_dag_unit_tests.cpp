/*
#include "../include/computational_dag.hpp"
#include "../include/functional.hpp"

#include <cassert>
#include <iostream>

namespace tinytorch {

ComputationalDAGNode<int> id_node([](const std::vector<Tensor<int>>& operands){
                                        assert(operands.size() == 1);
                                        return operands[0];
                                    });
ComputationalDAGNode<int> plus_2_node([](const std::vector<Tensor<int>>& operands){
                                        assert(operands.size() == 1);
                                        Tensor<int> c2 = constant<int>(operands[0].shape(), 2);
                                        return add(operands[0], c2);
                                    });
ComputationalDAGNode<int> times_3_node([](const std::vector<Tensor<int>>& operands){
                                        assert(operands.size() == 1);
                                        Tensor<int> c3 = constant<int>(operands[0].shape(), 3);
                                        return mul(operands[0], c3);
                                    });
ComputationalDAGNode<int> sum_node([](const std::vector<Tensor<int>> operands){
                                        assert(operands.size() == 2);
                                        return add(operands[0], operands[1]);
                                    });

void computationalDAGUnitTests(){
    // TO DO: improve framework and rewrite this
    ComputationalDAG<int> computational_dag;
    graph::NodeId id_node_id = computational_dag.addNode(id_node);
    std::cout << id_node_id << " ";
    graph::NodeId plus_2_node_id = computational_dag.addNode(plus_2_node);
    std::cout << plus_2_node_id << " ";
    graph::NodeId times_3_node_id = computational_dag.addNode(times_3_node);
    std::cout << times_3_node_id << " ";
    graph::NodeId sum_node_id = computational_dag.addNode(sum_node);
    std::cout << sum_node_id << std::endl;
    std::cout << "Size test 1: " << (computational_dag.size() == 4 ? "Passed" : "Failed") << std::endl;
    computational_dag.addEdge(id_node_id, plus_2_node_id);
    computational_dag.addEdge(id_node_id, times_3_node_id);
    computational_dag.addEdge(plus_2_node_id, sum_node_id);
    computational_dag.addEdge(times_3_node_id, sum_node_id);
    computational_dag.getEntryPoint() = id_node_id;
    computational_dag.getExitPoint() = sum_node_id;
    std::cout << "Adjacency lists" << std::endl;
    for(graph::NodeId id = 0; id < 4; ++id){
        for(auto s : computational_dag.getSuccessors(id)){
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Topo order" << std::endl;
    auto topo_order = computational_dag.topoOrder();
    for(auto id : topo_order){
        std::cout << id << " ";
    }
    std::cout << std::endl << std::endl;
    Tensor<int> input = iota<int>({2, 2});
    std::cout << "Input" << std::endl;
    std::cout << input;
    std::cout << "Result" << std::endl;
    std::cout << computational_dag.forward(input) << std::endl;
    std::cout << "Intermediate results" << std::endl;
    for(graph::NodeId id = 0; id < 4; ++id){
        std::cout << "Node id " << id << std::endl;
        std::cout << computational_dag.get(id).result_ << std::endl;
    }
}
    
} // namespace tinytorch
*/