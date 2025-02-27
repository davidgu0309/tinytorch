#include "computational_dag_unit_tests.cpp"
#include "graph_unit_tests.cpp"
#include "scalar_operation_unit_tests/run_scalar_operation_unit_tests.cpp"
#include "tensor_operation_unit_tests/run_tensor_operation_unit_tests.cpp"

#include <iostream>

int main() {

    // TODO: move to graph
    std::cout << "Graph unit tests" << std::endl;
    graphUnitTests();

    std::cout << std::endl;

    std::cout << "         SCALAR OPERATION UNIT TESTS" << std::endl << std::endl;
    scalarOperationUnitTests();

    std::cout << std::endl;
    
    std::cout << "         TENSOR OPERATION UNIT TESTS" << std::endl << std::endl;
    tensorOperationUnitTests();

    std::cout << std::endl;
    
    std::cout << "         COMPUTATIONAL DAG UNIT TESTS" << std::endl << std::endl;
    computationalDAGUnitTests();
    
    return 0;
    
}