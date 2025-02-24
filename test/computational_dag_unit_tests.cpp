#include "../include/computational_dag.hpp"
#include "../tensor_operation/include/all.hpp"
#include "../tensor/test/test_tensors.hpp"

using namespace tinytorch;

#define TYPE int
Shape SHAPE = {};

// Instantiate tensor operations
Addition<TYPE> addition;
Hadamard<TYPE> hadamard;
// Matmul<TYPE> matmul;

void computationalDAGUnitTests(){

    // Instantiate computational dag
    ComputationalDAG<TYPE> computational_dag;

    // Allocate input tensors
    InputId input_1 = computational_dag.addInput(SHAPE);
    InputId input_2 = computational_dag.addInput(SHAPE);

    // Allocate parameter tensors
    ParameterId w_1 = computational_dag.addParameter(SHAPE);

    // Define nodes (CYCLIC DEPENDENCIES ARE NOT SUPPORTED!)
    ComputationalDAGNode<TYPE> addition_node(addition, {{input_1, INPUT}, {input_2, INPUT}});
    graph::NodeId add_node = computational_dag.addNode(addition_node);
    ComputationalDAGNode<TYPE> multiplication_node(hadamard, {{w_1, PARAMETER}, {add_node, NODE}});
    graph::NodeId mul_node = computational_dag.addNode(multiplication_node);

    // Add edges TODO: do this automatically
    computational_dag.addEdge(add_node, mul_node);

    // Set entry point and exit point
    computational_dag.getEntryPoint() = add_node;
    computational_dag.getExitPoint() = mul_node;

    // Set inputs
    computational_dag.getInput(input_1) = ones<TYPE>(SHAPE);
    computational_dag.getInput(input_2) = iota<TYPE>(SHAPE);

    // Set parameters
    computational_dag.getParameter(w_1) = scalar_55;

    std::cout << "Forward test" << std::endl;
    std::cout << computational_dag.forward() << std::endl;

    std::cout << "Backward test" << std::endl;
    computational_dag.backward();
    std::cout << "mul_node.jacobi_[0] (param): " << computational_dag.get(mul_node).jacobi_[0] << std::endl;
    std::cout << "mul_node.jacobi_[1] (input): " << computational_dag.get(mul_node).jacobi_[1] << std::endl;
    std::cout << "add_node.jacobi_[0]: " << computational_dag.get(add_node).jacobi_[0] << std::endl;
    std::cout << "add_node.jacobi_[1]: " << computational_dag.get(add_node).jacobi_[1] << std::endl;
}

/*

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