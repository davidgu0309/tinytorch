#include "../include/scalar_operation.hpp"
#include "test.hpp"

namespace tinytorch {

    int scalarNegInt(const int x){
        return tinytorch::neg<int>(x);
    }

    void scalarOpUnitTests(){
        test::TestSuite<&scalarNegInt> negIntTests;
        negIntTests.addTest({0});
        negIntTests.addTest({1});
        negIntTests.run();
    }

} // namespace tinytorch