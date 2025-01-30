#include "computational_dag_unit_tests.cpp"
#include "functional_unit_tests.cpp"
#include "graph_unit_tests.cpp"
// #include "scalar_operation_unit_tests.cpp" // Besed on new framework

#include <iostream>

int main() {
    std::cout << "Running functional unit tests." << std::endl;
    tinytorch::functionalUnitTests();
    std::cout << "Running graph unit tests." << std::endl;
    tinytorch::graphUnitTests();
    std::cout << "Running computational DAG unit tests." << std::endl;
    tinytorch::computationalDAGUnitTests();
    /* Based on improved testing framework, doesn't work yet.
    std::cout << "Running scalar operation unit tests" << std::endl;
    tinytorch::scalarOpUnitTests();
    */
}