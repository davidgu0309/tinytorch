#include "computational_dag_unit_tests.cpp"
#include "functional_unit_tests.cpp"
#include "graph_unit_tests.cpp"

#include <iostream>

int main() {
    std::cout << "Running functional unit tests." << std::endl;
    tinytorch::functionalUnitTests();
    std::cout << "Running graph unit tests." << std::endl;
    tinytorch::graphUnitTests();
    std::cout << "Running computational DAG unit tests." << std::endl;
    tinytorch::computationalDAGUnitTests();
}