#include "test_functional.cpp"
#include <iostream>

int main() {
    std::cout << "Starting unit tests. \n";
    if (tinytorch::testAddTensors()) std::cout << "Test passed. \n";
    else std::cout << "Test failed \n";
}