# Simple build (alternative to CMake).
#   make          build the `dks` CLI
#   make test     build and run the correctness tests
#   make clean

CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
INC       = -Iinclude

.PHONY: all test clean

all: dks

dks: cli/dks_cli.cpp include/dks/dks.hpp
	$(CXX) $(CXXFLAGS) $(INC) cli/dks_cli.cpp -o dks

test_dks: tests/test_dks.cpp include/dks/dks.hpp
	$(CXX) $(CXXFLAGS) $(INC) tests/test_dks.cpp -o test_dks

test: test_dks
	./test_dks

clean:
	rm -f dks test_dks
