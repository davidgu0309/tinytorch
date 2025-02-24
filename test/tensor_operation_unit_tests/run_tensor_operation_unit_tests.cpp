#include "addition_unit_tests.cpp"
#include "hadamard_unit_tests.cpp"
#include "matmul_unit_tests.cpp"

void tensorOperationUnitTests() {
    addition_tests::runUnitTests();
    matmul_tests::runUnitTests();
    hadamard_tests::runUnitTests();
}