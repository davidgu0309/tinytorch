#include "../framework/test.hpp"
#include "../../include/scalar_operation/trigonometry.hpp"

using namespace test;
using namespace tinytorch;

namespace trigonometry_tests {

    double INF = std::numeric_limits<double>::infinity();

ScalarSin<double> sin_op;

double sin(const std::vector<double> operands){
    return sin_op(operands);
}

TestSuite<sin> sinUnitTests(){
    TestSuite<sin> tests("Sine operator tests");
    tests.addTest(ComparativeUnitTest<sin>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<sin>(std::vector<double>({M_PI / 2}), 1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<sin>(std::vector<double>({M_PI}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<sin>(std::vector<double>({3 * M_PI / 2}), -1.0, epsilon_comparison(1e-9)));
    return tests;
}

double sinBackward(const size_t input_idx, const std::vector<double> operands){
    return sin_op.backward(input_idx, operands);
}

TestSuite<sinBackward> sinBackwardUnitTests(){
    TestSuite<sinBackward> tests("Sine backward tests");
    tests.addTest(ComparativeUnitTest<sinBackward>({0, std::vector<double>({0.0})}, 1.0));
    tests.addTest(ComparativeUnitTest<sinBackward>({0, std::vector<double>({M_PI / 2})}, 0.0, epsilon_comparison(1e-9)));
    return tests;
}

ScalarCos<double> cos_op;

double cos(const std::vector<double> operands){
    return cos_op(operands);
}

TestSuite<cos> cosUnitTests(){
    TestSuite<cos> tests("Cosine operator tests");
    tests.addTest(ComparativeUnitTest<cos>(std::vector<double>({0.0}), 1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<cos>(std::vector<double>({M_PI / 2}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<cos>(std::vector<double>({M_PI}), -1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<cos>(std::vector<double>({3 * M_PI / 2}), 0.0, epsilon_comparison(1e-9)));
    return tests;
}

double cosBackward(const size_t input_idx, const std::vector<double> operands){
    return cos_op.backward(input_idx, operands);
}

TestSuite<cosBackward> cosBackwardUnitTests(){
    TestSuite<cosBackward> tests("Cosine backward tests");
    tests.addTest(ComparativeUnitTest<cosBackward>({0, std::vector<double>({0.0})}, 0.0));
    tests.addTest(ComparativeUnitTest<cosBackward>({0, std::vector<double>({M_PI / 2})}, -1.0, epsilon_comparison(1e-9)));
    return tests;
}

ScalarTan<double> tan_op;

double tan(const std::vector<double> operands){
    return tan_op(operands);
}

TestSuite<tan> tanUnitTests(){
    TestSuite<tan> tests("Tangent operator tests");
    tests.addTest(ComparativeUnitTest<tan>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<tan>(std::vector<double>({M_PI / 4}), 1.0, epsilon_comparison(1e-9)));
    // tests.addTest(ComparativeUnitTest<tan>(std::vector<double>({M_PI / 2}), INF, epsilon_comparison(1e-9)));
    // tests.addTest(ComparativeUnitTest<tan>(std::vector<double>({3 * M_PI / 2}), -INF, epsilon_comparison(1e-9)));
    return tests;
}

double tanBackward(const size_t input_idx, const std::vector<double> operands){
    return tan_op.backward(input_idx, operands);
}

TestSuite<tanBackward> tanBackwardUnitTests(){
    TestSuite<tanBackward> tests("Tangent backward tests");
    tests.addTest(ComparativeUnitTest<tanBackward>({0, std::vector<double>({0.0})}, 1.0));
    // tests.addTest(ComparativeUnitTest<tanBackward>({0, std::vector<double>({M_PI / 2})}, INF, epsilon_comparison(1e-9)));
    return tests;
}

void runUnitTests() {

    sinUnitTests().run();
    sinBackwardUnitTests().run();

    cosUnitTests().run();
    cosBackwardUnitTests().run();

    tanUnitTests().run();
    tanBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

