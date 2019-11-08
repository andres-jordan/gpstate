# define the C compiler to use
CC = g++

# define any compile-time flags
CFLAGS = -O3 -march=native -DNDEBUG
# define any directories containing header files other than /usr/include
#
INCLUDES = -I/usr/include/eigen3
# define the C+ source files
SRCS = src/benchmark.cpp  src/benchmark_ndsho.cpp  src/simulate_matern32.cpp  src/simulate_ndsho.cpp  src/simulate_sin.cpp  src/test_logl_dsho.cpp  src/test_logl_dsin.cpp  src/test_varf.cpp

BIN = $(patsubst src/%.cpp,%,$(SRCS))

all: $(BIN)

$(BIN): %: src/%.cpp include/*.h
	$(CC) $(INCLUDES) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(BIN)
