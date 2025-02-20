#include "../tensor/include/scalar_operation.hpp"
#include "test.hpp"

using namespace tensor;

int scalarNegInt(const int x){
    return neg<int>(x);
}

void scalarOpUnitTests(){
    test::TestSuite<&scalarNegInt> negIntTests;
    negIntTests.addTest({0});
    negIntTests.addTest({1});
    negIntTests.run();
}