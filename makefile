.PHONY: all test clean

all: src/util.cpp src/main.cpp
	g++ -o main_binary -g src/util.cpp src/main.cpp && ./main_binary

test: src/util.cpp test/run_unit_tests.cpp
	g++ -o test_binary -g src/util.cpp test/run_unit_tests.cpp && ./test_binary

clean:
	rm -f main_binary && rm -f test_binary

