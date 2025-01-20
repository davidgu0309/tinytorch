all: 
	g++ -o run -g src/utils.cpp src/main.cpp && ./run

test: 
	g++ -o test -g src/utils.cpp tests/run_unit_tests.cpp && ./test

clean:
	rm -f run && rm -f test

