# gpstate

Installation
=============

Preliminaries
---------------

You need to install celerite to run the benchmark.

To install only locally, follow the following instructions.

Download celerite

        git clone https://github.com/dfm/celerite.git

Install C++ headers

        cd celerite/cpp
        cmake .
        make

Now, back in the gpstate directory,
edit Makefile so that INCLUDES point to celerite.

Installation instructions
--------------------------

Installing is as simple as 
        
        make

Run benchmarks

        ./benchmark
        ./benchmark_ndsho
