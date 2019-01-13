#!/bin/bash

cmake -DCMAKE_C_COMPILER=/depot/gcc-6.2.0/bin/gcc -DCMAKE_CXX_COMPILER=/depot/gcc-6.2.0/bin/g++ -DCMAKE_BUILD_TYPE=Release && make -j 4
