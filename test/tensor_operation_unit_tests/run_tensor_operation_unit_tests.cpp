#include "activation_unit_tests.cpp"
#include "addition_unit_tests.cpp"
#include "hadamard_unit_tests.cpp"
#include "loss_unit_tests.cpp"
#include "matmul_unit_tests.cpp"

void tensorOperationUnitTests() {
    tensor_activation_tests::runUnitTests();
    tensor_addition_tests::runUnitTests();
    hadamard_tests::runUnitTests();
    tensor_loss_tests::runUnitTests();
    matmul_tests::runUnitTests();
}