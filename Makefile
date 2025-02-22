.PHONY: all test clean

all: 
	g++ --std=c++20 -o main_binary -g tensor/src/util.cpp src/main.cpp && ./main_binary

test:
	g++ --std=c++20 -o test_binary -g tensor/src/util.cpp test/run_unit_tests.cpp && ./test_binary

clean:
	rm -f main_binary && rm -f test_binary

