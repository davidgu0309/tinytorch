test: 
	g++ -o test -g src/functional.cpp src/utils.cpp tests/test_main.cpp

clean:
	rm -f run && rm -f test