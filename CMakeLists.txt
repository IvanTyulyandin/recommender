cmake_minimum_required(VERSION 3.6)
project(recommender)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -D_GLIBCXX_PARALLEL -pthread -Wall")

add_executable(recommender main.cpp recommender.cpp recommender.h crossValidation.h)
