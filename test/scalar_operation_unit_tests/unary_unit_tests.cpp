#include "../framework/test.hpp"
#include "../../include/scalar_operation/unary.hpp"

using namespace test;
using namespace tinytorch;

namespace unary_tests {

    double INF = std::numeric_limits<double>::infinity();

ScalarSgn<double> sgn_op;

double sgn(const std::vector<double> operands){
    return sgn_op(operands);
}

TestSuite<sgn> sgnUnitTests(){
    TestSuite<sgn> tests("Sign operator tests");
    tests.addTest(ComparativeUnitTest<sgn>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<sgn>(std::vector<double>({M_PI / 2}), 1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<sgn>(std::vector<double>({- M_PI}), -1.0, epsilon_comparison(1e-9)));
    return tests;
}

double sgnBackward(const size_t input_idx, const std::vector<double> operands){
    return sgn_op.backward(input_idx, operands);
}

TestSuite<sgnBackward> sgnBackwardUnitTests(){
    TestSuite<sgnBackward> tests("Sign backward tests");
    tests.addTest(ComparativeUnitTest<sgnBackward>({0, std::vector<double>({0.0})}, INF));
    tests.addTest(ComparativeUnitTest<sgnBackward>({0, std::vector<double>({M_PI / 2})}, 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<sgnBackward>({0, std::vector<double>({-M_PI / 2})}, 0.0, epsilon_comparison(1e-9)));
    return tests;
}

ScalarAbs<double> abs_op;

double abs(const std::vector<double> operands){
    return abs_op(operands);
}

TestSuite<abs> absUnitTests(){
    TestSuite<abs> tests("Absolute value operator tests");
    tests.addTest(ComparativeUnitTest<abs>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<abs>(std::vector<double>({M_PI / 2}), M_PI / 2, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<abs>(std::vector<double>({-M_PI}), M_PI, epsilon_comparison(1e-9)));
    return tests;
}

double absBackward(const size_t input_idx, const std::vector<double> operands){
    return abs_op.backward(input_idx, operands);
}

TestSuite<absBackward> absBackwardUnitTests(){
    TestSuite<absBackward> tests("Absolute value backward tests");
    tests.addTest(ComparativeUnitTest<absBackward>({0, std::vector<double>({0.0})}, 0.0));
    tests.addTest(ComparativeUnitTest<absBackward>({0, std::vector<double>({M_PI / 2})}, 1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<absBackward>({0, std::vector<double>({-M_PI / 2})}, -1.0, epsilon_comparison(1e-9)));
    return tests;
}

ScalarNeg<double> neg_op;

double neg(const std::vector<double> operands){
    return neg_op(operands);
}

TestSuite<neg> negUnitTests(){
    TestSuite<neg> tests("Negation operator tests");
    tests.addTest(ComparativeUnitTest<neg>(std::vector<double>({0.0}), 0.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<neg>(std::vector<double>({M_PI / 2}), -M_PI / 2, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<neg>(std::vector<double>({-M_PI}), M_PI, epsilon_comparison(1e-9)));
    return tests;
}

double negBackward(const size_t input_idx, const std::vector<double> operands){
    return neg_op.backward(input_idx, operands);
}

TestSuite<negBackward> negBackwardUnitTests(){
    TestSuite<negBackward> tests("Negation backward tests");
    tests.addTest(ComparativeUnitTest<negBackward>({0, std::vector<double>({0.0})}, -1.0));
    tests.addTest(ComparativeUnitTest<negBackward>({0, std::vector<double>({M_PI / 2})}, -1.0, epsilon_comparison(1e-9)));
    tests.addTest(ComparativeUnitTest<negBackward>({0, std::vector<double>({-M_PI / 2})}, -1.0, epsilon_comparison(1e-9)));
    return tests;
}

void runUnitTests() {

    sgnUnitTests().run();
    sgnBackwardUnitTests().run();

    absUnitTests().run();
    absBackwardUnitTests().run();

    negUnitTests().run();
    negBackwardUnitTests().run();

    // TODO: manual tests
    // TODO: randomized tests
}

} // namespace addition

