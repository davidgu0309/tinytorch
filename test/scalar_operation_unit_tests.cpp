#include "../include/scalar_operation.hpp"
#include "test.hpp"

namespace tinytorch {

    int scalarNegInt(const int x){
        return tinytorch::neg<int>(x);
    }

    void scalarOpUnitTests(){
        test::TestSuite<decltype(scalarNegInt)> negIntTests;
        negIntTests.addTest({{0}, 0});
        negIntTests.addTest({{-1}, 1});
        negIntTests.run();
    }

} // namespace tinytorch