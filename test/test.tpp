namespace test{
    
    template<typename T>
    void consoleLog(const T& result, const T& correct){
        bool passed = result == correct;
        std::cout << (passed ? "Passed" : "Failed") << std::endl;
        if(!passed){
            std::cout << "Result:" << std::endl;
            std::cout << result << std::endl;
            std::cout << "Correct:" << std::endl;
            std::cout << correct << std::endl;
        }
    }

    template<typename FunctionToTest>
    bool UnitTest<FunctionToTest>::run() const {
        ReturnType result = FunctionToTest(operands);
        if(result != correct_result) consoleLog(result, correct_result);
        return result == correct_result;
    }

    template<typename FunctionToTest>
    void TestSuite<FunctionToTest>::addTest(UnitTest<FunctionToTest> unit_test) {
        unit_tests.push_back(unit_test);
    };

    template<typename FunctionToTest>
    void TestSuite<FunctionToTest>::run() const {
        for(const UnitTest<FunctionToTest>& test : unit_tests){
            test.run();
        }
    }

}   // namespace test