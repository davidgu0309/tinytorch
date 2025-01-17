#include "functional_unit_tests.cpp"
#include <iostream>

int main() {
    std::cout << "Starting unit tests. \n";
    if (tinytorch::additionTest1()) std::cout << "Test passed. \n";
    else std::cout << "Test failed \n";
}