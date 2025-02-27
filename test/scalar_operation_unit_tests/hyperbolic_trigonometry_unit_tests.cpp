#include "../framework/test.hpp"
#include "../../include/scalar_operation/hyperbolic_trigonometry.hpp"

using namespace test;
using namespace tinytorch;

namespace hyperbolic_trigonometry_tests {

double INF = std::numeric_limits<double>::infinity();

ScalarSinH<double> sinh_op;

double sinh(const std::vector<double> operands){
    return sinh_op(operands);
}

TestSuite<sinh> sinhUnitTests(){
    TestSuite<sinh> tests("Hyperbolic sine operator tests");
    tests.addTest(ComparativeUnitTest<sinh>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    return tests;
}

double sinhBackward(const size_t input_idx, const std::vector<double> operands){
    return sinh_op.backward(input_idx, operands);
}

TestSuite<sinhBackward> sinhBackwardUnitTests(){
    TestSuite<sinhBackward> tests("Hyperbolic sine backward tests");
    tests.addTest(ComparativeUnitTest<sinhBackward>({0, std::vector<double>({0.0})}, 1.0));
    return tests;
}

ScalarCosH<double> cosh_op;

double cosh(const std::vector<double> operands){
    return cosh_op(operands);
}

TestSuite<cosh> coshUnitTests(){
    TestSuite<cosh> tests("Hyperbolic cosine operator tests");
    tests.addTest(ComparativeUnitTest<cosh>(std::vector<double>({0.0}), 1.0, epsilon_comparison(1e-9)));
    return tests;
}

double coshBackward(const size_t input_idx, const std::vector<double> operands){
    return cosh_op.backward(input_idx, operands);
}

TestSuite<coshBackward> coshBackwardUnitTests(){
    TestSuite<coshBackward> tests("Hyperbolic cosine backward tests");
    tests.addTest(ComparativeUnitTest<coshBackward>({0, std::vector<double>({0.0})}, 0.0));
    return tests;
}

ScalarTanH<double> tanh_op;

double tanh(const std::vector<double> operands){
    return tanh_op(operands);
}

TestSuite<tanh> tanhUnitTests(){
    TestSuite<tanh> tests("Hyperbolic tangent operator tests");
    tests.addTest(ComparativeUnitTest<tanh>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<tanh>(std::vector<double>({-INF}), -1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<tanh>(std::vector<double>({INF}), 1.0, epsilon_comparison(1e-9)));
    return tests;
}

double tanhBackward(const size_t input_idx, const std::vector<double> operands){
    return tanh_op.backward(input_idx, operands);
}

TestSuite<tanhBackward> tanhBackwardUnitTests(){
    TestSuite<tanhBackward> tests("Hyperbolic tangent backward tests");
    tests.addTest(ComparativeUnitTest<tanhBackward>({0, std::vector<double>({0.0})}, 1.0));
    tests.addTest(ComparativeUnitTest<tanhBackward>({0, std::vector<double>({INF})}, 0.0));
    tests.addTest(ComparativeUnitTest<tanhBackward>({0, std::vector<double>({-INF})}, 0.0));
    return tests;
}

void runUnitTests() {

    sinhUnitTests().run();
    sinhBackwardUnitTests().run();

    coshUnitTests().run();
    coshBackwardUnitTests().run();

    tanhUnitTests().run();
    tanhBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

