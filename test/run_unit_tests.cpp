#include "computational_dag_unit_tests.cpp"
#include "graph_unit_tests.cpp"
#include "tensor_operation_unit_tests/run_tensor_operation_unit_tests.cpp"

#include <iostream>

int main() {
    std::cout << "Running graph unit tests." << std::endl;
    graphUnitTests();
    std::cout << "Running tensor operation unit tests." << std::endl;
    tensorOperationUnitTests();
    // TO DO: write new computational DAG unit tests
    // std::cout << "Running computational DAG unit tests." << std::endl;
    // tinytorch::computationalDAGUnitTests();
    // Based on improved testing framework, doesn't work yet.
    // std::cout << "Running scalar operation unit tests" << std::endl;
    // tinytorch::scalarOpUnitTests();
}