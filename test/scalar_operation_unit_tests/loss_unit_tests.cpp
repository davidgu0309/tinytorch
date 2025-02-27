#include "../framework/test.hpp"
#include "../../include/scalar_operation/loss.hpp"

using namespace test;
using namespace tinytorch;

namespace loss_tests {

double INF = std::numeric_limits<double>::infinity();

ScalarLp<double> lp_op;

double lp(const std::vector<double> operands){
    return lp_op(operands);
}

TestSuite<lp> lpUnitTests(){
    TestSuite<lp> tests("Square loss operator tests");
    tests.addTest(ComparativeUnitTest<lp>(std::vector<double>({4.3, 0.3}), 16.0));
    return tests;
}

double lpBackward(const size_t input_idx, const std::vector<double> operands){
    return lp_op.backward(input_idx, operands);
}

TestSuite<lpBackward> lpBackwardUnitTests(){
    TestSuite<lpBackward> tests("Square loss backward tests");
    tests.addTest(ComparativeUnitTest<lpBackward>({0, std::vector<double>({3.0, 0.0})}, 6.0));
    tests.addTest(ComparativeUnitTest<lpBackward>({1, std::vector<double>({3.0, 0.0})}, -6.0));
    return tests;
}

void runUnitTests() {

    lpUnitTests().run();
    lpBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

