all: 
	g++ -o run -g src/utils.cpp src/main.cpp

test: 
	g++ -o test -g src/utils.cpp tests/run_unit_tests.cpp

clean:
	rm -f run && rm -f test

