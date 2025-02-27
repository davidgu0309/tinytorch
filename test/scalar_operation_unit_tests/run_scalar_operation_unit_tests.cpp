#include "activation_unit_tests.cpp"
#include "aggregation_unit_tests.cpp"
#include "hyperbolic_trigonometry_unit_tests.cpp"
#include "trigonometry_unit_tests.cpp"

void scalarOperationUnitTests() {
    activation_tests::runUnitTests();
    aggregation_tests::runUnitTests();
    hyperbolic_trigonometry_tests::runUnitTests();
    trigonometry_tests::runUnitTests();
}