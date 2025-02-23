#include "computational_dag_unit_tests.cpp"
#include "graph_unit_tests.cpp"
#include "tensor_operation_unit_tests/run_tensor_operation_unit_tests.cpp"

#include <iostream>

int main() {

    std::cout << "Running graph unit tests." << std::endl;
    graphUnitTests();
    
    std::cout << "Running tensor operation unit tests." << std::endl;
    tensorOperationUnitTests();
    
    std::cout << "Running computational DAG unit tests." << std::endl;
    computationalDAGUnitTests();

    return 0;
    
}