.PHONY: all test clean

all: tensor/src/util.cpp src/main.cpp
	g++ --std=c++17 -o main_binary -g src/util.cpp src/main.cpp && ./main_binary

test: tensor/src/util.cpp test/run_unit_tests.cpp
	g++ --std=c++17 -o test_binary -g src/util.cpp test/run_unit_tests.cpp && ./test_binary

clean:
	rm -f main_binary && rm -f test_binary

